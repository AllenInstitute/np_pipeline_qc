import argparse
import concurrent.futures
import doctest
import functools
import pathlib
from typing import Generator

import matplotlib.pyplot as plt
import numba
import numpy as np
import numpy.typing as npt
import np_logging

logger = np_logging.getLogger(__name__)


SAMPLE_RATE = 30000.0

GLOB_AP_DIR = '**/continuous/Neuropix-PXI-???.0'
"""Glob pattern for finding AP directories in a parent directory containing Kilosort
output subfolders."""

BATCH_SIZE = 50000
"""Number of spikes to process at a time. This is to avoid memory errors when
processing."""


def generate_and_save_spike_depths_single_probe(continuous_AP_dir) -> None:
    spike_depths = get_spike_depths_single_probe(continuous_AP_dir)
    np.save(continuous_AP_dir / 'spike_depths.npy', spike_depths)


def save_all_spike_depths(parent_dir: pathlib.Path) -> None:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(
            generate_and_save_spike_depths_single_probe,
            sorted_continuous_AP_dirs(parent_dir),
        )


@numba.njit
def calculate_spike_depths(
    sparse_features,
    sparse_features_ind,
    spike_templates,
    spike_times,
    channel_positions,
) -> npt.NDArray:
    num_spikes = spike_times.shape[0]
    spike_depths = np.empty_like(spike_times)

    channel_idx = sparse_features_ind[spike_templates].astype(np.uint32)
    features = np.abs(
        np.maximum(sparse_features[:, :, 0], 0) ** 2
    )  # takes only positive values into account
    ypos = channel_positions[channel_idx, 1]
    with np.errstate(divide='ignore'):
        return np.sum(
            np.transpose(ypos * features) / np.sum(features, axis=1),
            axis=0,
        )

    # # if memory becomes an issue, use this method instead:
    # return low_mem_calc(
    #         sparse_features,
    #         sparse_features_ind,
    #         spike_templates,
    #         spike_times,
    #         channel_positions,
    #     )


def low_mem_calc(
    sparse_features,
    sparse_features_ind,
    spike_templates,
    spike_times,
    channel_positions,
) -> npt.NDArray:
    
    num_spikes = spike_times.shape[0]
    spike_depths = np.empty_like(spike_times)
    
    c = 0
    while c < num_spikes:

        idx = np.arange(c, min(c + BATCH_SIZE, num_spikes))
        logger.debug('Processing spikes %d to %d', c, c + idx.shape[0])

        # take only first component
        features = sparse_features[idx, :, 0]
        features = (
            np.maximum(features, 0) ** 2
        )  # takes only positive values into account

        ichannels = sparse_features_ind[spike_templates[idx]].astype(np.uint32)
        # features = np.square(self.sparse_features.data[idx, :, 0])
        # ichannels = self.sparse_features.cols[self.spike_templates[idx]].astype(np.int64)
        ypos = channel_positions[ichannels, 1]
        # ypos = ypos[:, 0, :]

        with np.errstate(divide='ignore'):
            spike_depths[idx] = np.sum(
                np.transpose(ypos * features) / np.sum(features, axis=1),
                axis=0,
            )

        c += BATCH_SIZE

    return spike_depths


def get_spike_depths_single_probe(
    continuous_AP_dir: pathlib.Path,
) -> npt.NDArray:
    """Generate spike depths for a given Kilosort output dir."""

    logger.debug('Generating spike depths for %s', continuous_AP_dir)

    logger.debug('Loading arrays...')
    arrays_for_spike_depth_calc = get_arrays_for_spike_depth_calc(
        continuous_AP_dir
    )

    logger.debug('Calculating spike depths...')
    return calculate_spike_depths(*arrays_for_spike_depth_calc)

@functools.lru_cache(maxsize=1)
def get_arrays_for_spike_depth_calc(
    continuous_AP_dir,
) -> tuple[npt.NDArray, ...]:
    fns = (
        get_sparse_features,
        get_sparse_features_ind,
        get_spike_templates,
        get_spike_times,
        get_channel_positions,
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=6 * len(fns)
    ) as executor:
        futures = tuple(executor.submit(fn, continuous_AP_dir) for fn in fns)
    return tuple(future.result() for future in futures)


def get_sparse_features_ind(continuous_AP_dir) -> npt.NDArray:
    return np.load(continuous_AP_dir / 'pc_feature_ind.npy', mmap_mode='r')


def get_sparse_features(continuous_AP_dir) -> npt.NDArray:
    pc_features: npt.NDArray = np.load(
        continuous_AP_dir / 'pc_features.npy', mmap_mode='r'
    )
    sparse_features = pc_features.squeeze().transpose((0, 2, 1))
    logger.debug('Loaded sparse features, shape: %s', sparse_features.shape)
    return sparse_features


def get_spike_templates(continuous_AP_dir) -> npt.NDArray:
    spike_templates: npt.NDArray = np.load(
        continuous_AP_dir / 'spike_templates.npy', mmap_mode='r'
    )[:, 0]
    logger.debug('Loaded templates, shape: %s', spike_templates.shape)
    return spike_templates


def get_spike_times(continuous_AP_dir) -> npt.NDArray:
    spike_times: npt.NDArray = np.load(
        continuous_AP_dir / 'spike_times.npy', mmap_mode='r'
    )[:, 0]
    spike_times = spike_times / SAMPLE_RATE
    logger.debug('Loaded spike times, shape: %s', spike_times.shape)
    return spike_times


def get_channel_positions(continuous_AP_dir) -> npt.NDArray:
    channel_positions: npt.NDArray = np.load(
        continuous_AP_dir / 'channel_positions.npy', mmap_mode='r'
    )
    logger.debug(
        'Loaded channel positions, shape: %s', channel_positions.shape
    )
    return channel_positions


def sorted_continuous_AP_dirs(
    parent: str | pathlib.Path,
) -> Generator[pathlib.Path, None, None]:
    """Generator yielding any sorted AP directories within any parent directory.

    >>> next(sorted_continuous_AP_dirs('//allen/programs/mindscope/production/variability/prod0/specimen_1153809254/ecephys_session_1176583610/1184386053/1176693482_probeA/continuous/Neuropix-PXI-100.0')).as_posix()
    '//allen/programs/mindscope/production/variability/prod0/specimen_1153809254/ecephys_session_1176583610/1184386053/1176693482_probeA/continuous/Neuropix-PXI-100.0'
    >>> next(sorted_continuous_AP_dirs('//allen/programs/mindscope/production/variability/prod0/specimen_1153809254/ecephys_session_1176583610/1184386053')).as_posix()
    '//allen/programs/mindscope/production/variability/prod0/specimen_1153809254/ecephys_session_1176583610/1184386053/1176693482_probeA/continuous/Neuropix-PXI-100.0'
    """
    parent = pathlib.Path(parent)
    if parent.match(GLOB_AP_DIR):
        yield parent
    yield from parent.glob(GLOB_AP_DIR)
    
def plot_driftmap_all_probes(session_dir_or_probe_dir: pathlib.Path) -> plt.Figure:
    for continuous_AP_dir in sorted_continuous_AP_dirs(session_dir_or_probe_dir):
        plot_driftmap_single_probe(continuous_AP_dir)
    plt.show()

def plot_driftmap_single_probe(continuous_AP_dir: pathlib.Path) -> plt.Figure:
    """Looks for `spike_depths.npy` and `spike_times.npy` in the given directory."""
    
    depths_file, times_file = (files := tuple(continuous_AP_dir / f'spike_{_}.npy' for _ in ('depths', 'times')))
    
    if not times_file.exists():
        raise FileNotFoundError(f'Could not find {times_file}')
    if not depths_file.exists():
        generate_and_save_spike_depths_single_probe(continuous_AP_dir)
        
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(files)*6
    ) as executor:
        futures = tuple(executor.submit(np.load, file, mmap_mode='r') for file in files)
    spike_depths, spike_times = [future.result() for future in futures]
    
    spike_times = spike_times.squeeze() / SAMPLE_RATE
    spike_depths = spike_depths.squeeze()
    
    # time-depth 2D histogram
    num_time_bins = 1000
    num_depth_bins = 400

    time_bins = np.linspace(0, spike_times.max(), num_time_bins)
    depth_bins = np.linspace(0, np.nanmax(spike_depths), num_depth_bins)

    time_bin_size_sec = np.mean(np.diff(time_bins))
    
    spk_counts, spk_edges, depth_edges = np.histogram2d(spike_times, spike_depths, bins=[time_bins, depth_bins])
    spk_rates = spk_counts / time_bin_size_sec
    spk_edges = spk_edges[:-1]
    depth_edges = depth_edges[:-1]
    
    # canvas setup
    fig = plt.figure(figsize=(16, 8))
    grid = plt.GridSpec(12, 12)

    ax_main = plt.subplot(grid[:, 0:10])
    # ax_cbar = plt.subplot(grid[0, 0:10])
    ax_spkcount = plt.subplot(grid[:, 10:])

    # -- plot main --
    im = ax_main.imshow(spk_rates.T, aspect='auto', cmap='gray_r',
                        extent=[time_bins[0], time_bins[-1], depth_bins[-1], depth_bins[0]])
    # cosmetic
    ax_main.invert_yaxis()
    ax_main.set_xlabel('Time (sec)')
    ax_main.set_ylabel('Distance from tip sites (um)')
    ax_main.set_ylim(depth_edges[0], depth_edges[-1])
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['top'].set_visible(False)

    cb = fig.colorbar(im, cax=ax_cbar, orientation='horizontal')
    cb.outline.set_visible(False)
    cb.ax.xaxis.tick_top()
    cb.set_label('Firing rate (Hz)')
    cb.ax.xaxis.set_label_position('top')

    # -- plot spikecount --
    ax_spkcount.plot(spk_counts.sum(axis=0) / 10e3, depth_edges, 'k')
    ax_spkcount.set_xlabel('Spike count (x$10^3$)')
    ax_spkcount.set_yticks([])
    ax_spkcount.set_ylim(depth_edges[0], depth_edges[-1])

    ax_spkcount.spines['right'].set_visible(False)
    ax_spkcount.spines['top'].set_visible(False)
    ax_spkcount.spines['bottom'].set_visible(False)
    ax_spkcount.spines['left'].set_visible(False)
    
    fig.suptitle(f'{continuous_AP_dir}')
    
    return fig

def main() -> None:

    np_logging.getLogger()

    parser = argparse.ArgumentParser(
        prog=__name__,
        usage="""
Supply a path to either:
- a Kilosort 2 continuous/AP output directory 
    (e.g. a directory ending `continuous/Neuropix-PXI-???.0` for a single sorted probe)
- any parent of continuous/AP directories 
    (e.g. a session directory containing multiple sorted probe folders)

Within the path supplied, and any subfolders thereof, we'll search for continuous/AP
dirs and use the following files:
    - pc_features.npy
    - pc_feature_ind.npy
    - spike_templates.npy
    - spike_times.npy
    - channel_positions.npy
    
In each continuous/AP dir, a spike_depths.npy file will be created.
""",
        description='Generates spike_depths.npy for all AP output directory subfolders for a given path.',
    )
    parser.add_help = True
    parser.add_argument(
        'session_dir_or_probe_dir',
        type=pathlib.Path,
        help='Path to a directory containing one or more subfolders of AP-band output files from Kilosort 2.',
    )
    session_dir_or_probe_dir = parser.parse_args().session_dir_or_probe_dir
    save_all_spike_depths(session_dir_or_probe_dir)


if __name__ == '__main__':
    
    logger = np_logging.getLogger()

    TEST_DIR = pathlib.Path(
        '//allen/programs/mindscope/workgroups/np-exp/1256079153_661728_20230321'
    )
    TEST = 2
    match TEST:
        
        case 0: # no test
            main()
            
        case 2: # test plotting
            plot_driftmap_all_probes(TEST_DIR) 
        
        case 1: # test generating spike_depths
            for continuous_AP_dir in sorted_continuous_AP_dirs(TEST_DIR):
                (TEST_DIR / 'spike_depths.npy').unlink(missing_ok=True)

            save_all_spike_depths(TEST_DIR)

            for continuous_AP_dir in sorted_continuous_AP_dirs(TEST_DIR):
                assert (continuous_AP_dir / 'spike_depths.npy').exists() and (
                    continuous_AP_dir / 'spike_depths.npy'
                ).stat().st_size > 0

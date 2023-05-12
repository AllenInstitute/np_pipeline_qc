from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import doctest
import functools
import logging
import os
import pathlib
import re
import tempfile
from typing import Generator, Literal, Optional

import matplotlib.pyplot as plt
import np_logging
import numba
import numpy as np
import numpy.typing as npt

from np_pipeline_qc import utils

logger = logging.getLogger(__name__)


SAMPLE_RATE = 30000.0

BATCH_SIZE = 50000
"""Number of spikes to process at a time. This is to avoid memory errors when
processing."""


def spike_depths_npy(continuous_AP_dir: pathlib.Path) -> pathlib.Path:
    """Path to the spike depths file for a single probe."""
    return continuous_AP_dir / 'spike_depths.npy'


def generate_and_save_spike_depth_array_single_probe(continuous_AP_dir) -> None:
    spike_depths = get_spike_depth_array_single_probe(continuous_AP_dir)
    np.save(spike_depths_npy(continuous_AP_dir), spike_depths)


def save_spike_depth_array_all_probes(parent_dir: pathlib.Path, skip_existing: bool = True) -> None:
    futures: list[concurrent.futures.Future] = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for continuous_AP_dir in utils.sorted_continuous_AP_dirs(parent_dir):
            probe_letter = utils.get_probe_letter_group_from_path(continuous_AP_dir)
            spike_depths_path = spike_depths_npy(continuous_AP_dir)
            if skip_existing and spike_depths_path.exists():
                logger.info(f'Skipping generation: {spike_depths_path.relative_to(parent_dir)} already exists')
                continue
            logger.debug(f'Generating {spike_depths_path.relative_to(parent_dir)}')
            futures.append(executor.submit(generate_and_save_spike_depth_array_single_probe, continuous_AP_dir))
    for future in futures:
        if future.exception() is not None:
            raise future.exception()

# @numba.njit(error_model='numpy')
# @numba.jit
def calculate_spike_depths(
    sparse_features = np.array([[[]]], dtype=np.float32),
    sparse_features_ind = np.array([[]], dtype=np.uint32),
    spike_templates = np.array([], dtype=np.uint32),
    spike_times = np.array([], dtype=np.float64),
    channel_positions = np.array([[]], dtype=np.float64),
) -> npt.NDArray:
    
    channel_idx = np.array([[]], dtype=np.uint32)
    ypos = np.array([[]], dtype=np.float64)
    features = np.array([[]], dtype=np.float32)
    
    channel_idx = sparse_features_ind[spike_templates, :].astype(np.uint32)
    ypos = channel_positions[:, 1][channel_idx]
    features = np.maximum(sparse_features[:, :, 0], 0) ** 2 # takes only +ve values into account
    # with np.errstate(divide='ignore'):
    return np.sum(
        np.transpose(ypos * features) / np.sum(features, axis=1),
        axis=0,
    )

def low_mem_calc_spike_depths(
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


def get_spike_depth_array_single_probe(
    continuous_AP_dir: pathlib.Path,
) -> npt.NDArray:
    """Generate spike depths for a given Kilosort output dir."""

    arrays_for_spike_depth_calc = get_arrays_for_spike_depth_calc(
        continuous_AP_dir
    )
    return calculate_spike_depths(*arrays_for_spike_depth_calc)
    # with contextlib.suppress(Exception):
    #     return calculate_spike_depths(*arrays_for_spike_depth_calc)
    # return low_mem_calc_spike_depths(*arrays_for_spike_depth_calc)

# @functools.lru_cache(maxsize=1)
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


def get_sparse_features_ind(continuous_AP_dir) -> npt.NDArray[np.uint32]:
    return np.load(continuous_AP_dir / 'pc_feature_ind.npy', mmap_mode='r')


def get_sparse_features(continuous_AP_dir) -> npt.NDArray[np.float32]:
    pc_features: npt.NDArray = np.load(
        continuous_AP_dir / 'pc_features.npy', mmap_mode='r'
    )
    sparse_features = pc_features.squeeze().transpose((0, 2, 1))
    logger.debug('Loaded sparse features, shape: %s', sparse_features.shape)
    return sparse_features


def get_spike_templates(continuous_AP_dir) -> npt.NDArray[np.uint32]:
    spike_templates: npt.NDArray = np.load(
        continuous_AP_dir / 'spike_templates.npy', mmap_mode='r'
    )[:, 0]
    logger.debug('Loaded templates, shape: %s', spike_templates.shape)
    return spike_templates


def get_spike_times(continuous_AP_dir) -> npt.NDArray[np.float64]:
    spike_times: npt.NDArray = np.load(
        continuous_AP_dir / 'spike_times.npy', mmap_mode='r'
    )[:, 0]
    spike_times = spike_times / SAMPLE_RATE
    logger.debug('Loaded spike times, shape: %s', spike_times.shape)
    return spike_times


def get_channel_positions(continuous_AP_dir) -> npt.NDArray[np.float64]:
    channel_positions: npt.NDArray = np.load(
        continuous_AP_dir / 'channel_positions.npy', mmap_mode='r'
    )
    logger.debug(
        'Loaded channel positions, shape: %s', channel_positions.shape
    )
    return channel_positions


def show_spike_depth_map_all_probes(session_dir_or_probe_dir: pathlib.Path) -> None:
    for continuous_AP_dir in utils.sorted_continuous_AP_dirs(session_dir_or_probe_dir):
        plot_spike_depth_map_single_probe(continuous_AP_dir)
    plt.show()

def save_spike_depth_map_all_probes(
    session_dir_or_probe_dir: str | pathlib.Path,
    plot_save_dir: str | pathlib.Path,
    prefix: str = '',
    ) -> None:

    session_dir_or_probe_dir = pathlib.Path(session_dir_or_probe_dir)

    plot_save_dir = pathlib.Path(plot_save_dir)
    plot_save_dir.mkdir(exist_ok=True, parents=True)

    save_spike_depth_array_all_probes(session_dir_or_probe_dir, skip_existing=True)
    
    prefix += '_' if (prefix and not prefix.endswith('_')) else ''
    for _ in utils.sorted_continuous_AP_dirs(session_dir_or_probe_dir):
        probe = utils.get_probe_letter_group_from_path(_)
        fig = plot_spike_depth_map_single_probe(_)
        fig.savefig(plot_save_dir / f'{prefix}Probe_{probe}_spike_depth_map.png')

        
def plot_spike_depth_map_single_probe(continuous_AP_dir: pathlib.Path) -> plt.Figure:
    """Looks for `spike_depths.npy` and `spike_times.npy` in the given directory."""
    
    depths_file, times_file = (files := tuple(continuous_AP_dir / f'spike_{_}.npy' for _ in ('depths', 'times')))
    
    if not times_file.exists():
        raise FileNotFoundError(f'Could not find {times_file}')
    if not depths_file.exists():
        generate_and_save_spike_depth_array_single_probe(continuous_AP_dir)
        
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(files)*6
    ) as executor:
        futures = tuple(executor.submit(np.load, file) for file in files)
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
    ax_main.grid(False)
    
    # cb = fig.colorbar(im, cax=ax_cbar, orientation='horizontal')
    # cb.outline.set_visible(False)
    # cb.ax.xaxis.tick_top()
    # cb.set_label('Firing rate (Hz)')
    # cb.ax.xaxis.set_label_position('top')

    # -- plot spikecount --
    ax_spkcount.plot(spk_counts.sum(axis=0) / time_bins[-1], depth_edges, 'k')
    ax_spkcount.set_xlabel('Mean activity (spikes / sec)')
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
    save_spike_depth_array_all_probes(session_dir_or_probe_dir, skip_existing=True)


if __name__ == '__main__':
    
    # logger = np_logging.getLogger()

    TEST_DIR = pathlib.Path(
        '//allen/programs/mindscope/workgroups/np-exp/1256079153_661728_20230321'
    )
    TEST = 2
    if TEST == 0: # no test
            main()
        
    if TEST == 1: # doctests
            doctest.testmod(verbose=True)
            
    if TEST == 2: # test generating spike_depths.npy files
            
            # remove existing files
            for continuous_AP_dir in utils.sorted_continuous_AP_dirs(TEST_DIR):
                spike_depths_npy(continuous_AP_dir).unlink(missing_ok=True)

            # regenerate
            save_spike_depth_array_all_probes(TEST_DIR, skip_existing=False)

            # assert new files exist
            for continuous_AP_dir in utils.sorted_continuous_AP_dirs(TEST_DIR):
                assert (
                    spike_depths_npy(continuous_AP_dir).exists() 
                    and spike_depths_npy(continuous_AP_dir).stat().st_size > 0
                ), f'Failed to create {spike_depths_npy(continuous_AP_dir)}'
            print('All spike_depths.npy files created successfully.')
            
    if TEST == 3: # test plotting
            show_spike_depth_map_all_probes(TEST_DIR)
        
    if TEST == 4: # test saving plots
            with tempfile.TemporaryDirectory() as temp:
                save_spike_depth_map_all_probes(TEST_DIR, temp)
                
                logger.info('Testing spike depth map save. Created %s', [_.name for _ in pathlib.Path(temp).iterdir()])
                
                os.startfile(temp)
                pause = input('Press enter to continue and delete temp dir...')
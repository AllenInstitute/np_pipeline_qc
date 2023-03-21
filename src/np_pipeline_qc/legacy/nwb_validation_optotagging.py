import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from allensdk.brain_observatory.ecephys.behavior_ecephys_session import (
    BehaviorEcephysSession,
)
from matplotlib import pyplot as plt
from numba import njit
from pynwb import NWBHDF5IO


@njit
def makePSTH_numba(
    spikes,
    startTimes,
    windowDur,
    binSize=0.001,
    convolution_kernel=0.05,
    avg=True,
):
    spikes = spikes.flatten()
    startTimes = startTimes - convolution_kernel / 2
    windowDur = windowDur + convolution_kernel
    bins = np.arange(0, windowDur + binSize, binSize)
    convkernel = np.ones(int(convolution_kernel / binSize))
    counts = np.zeros(bins.size - 1)
    for i, start in enumerate(startTimes):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start + windowDur)
        counts = (
            counts + np.histogram(spikes[startInd:endInd] - start, bins)[0]
        )

    counts = counts / startTimes.size
    counts = np.convolve(counts, convkernel) / (binSize * convkernel.size)
    return (
        counts[convkernel.size - 1 : -convkernel.size],
        bins[: -convkernel.size - 1],
    )


# for inwb, nwb_path in enumerate(nwb_paths):


def opto_validation(nwb_path):
    with NWBHDF5IO(nwb_path, 'r', load_namespaces=True) as nwb_io:
        session = BehaviorEcephysSession.from_nwb(nwbfile=nwb_io.read())

    print('processing file {}'.format(nwb_path))

    channels = session.get_channels()

    units = session.get_units()
    good_unit_filter = (
        (units['snr'] > 1)
        & (units['isi_violations'] < 1)
        & (units['firing_rate'] > 0.1)
    )
    units = units.loc[good_unit_filter]
    unitchannels = units.merge(
        channels, left_on='peak_channel_id', right_index=True
    )
    unitchannels = unitchannels.sort_values(
        'probe_vertical_position', ascending=False
    )

    spike_times = session.spike_times

    opto_table = session.optotagging_table

    durations = opto_table.duration.unique()
    levels = opto_table.level.unique()

    sessionID = session.metadata['ecephys_session_id']

    fig, ax = plt.subplots(len(levels), len(durations))
    fig.set_size_inches([16, 10])
    fig.suptitle(str(sessionID) + '_' + session.metadata['full_genotype'])
    for idur, duration in enumerate(durations):
        for il, level in enumerate(levels):
            opto_times = opto_table.loc[
                (opto_table['duration'] == duration)
                & (opto_table['level'] == level)
            ]['start_time'].values
            all_resp = []
            for iu, unit in unitchannels.iterrows():
                sts = spike_times[iu]

                if duration < 0.1:
                    resp = makePSTH_numba(
                        sts,
                        opto_times - 0.1,
                        0.2,
                        binSize=0.001,
                        convolution_kernel=0.002,
                    )[0]
                    resp = resp - np.mean(resp[:99])

                if duration > 0.1:
                    resp = makePSTH_numba(
                        sts,
                        opto_times - 0.25,
                        1.5,
                        binSize=0.001,
                        convolution_kernel=0.01,
                    )[0]
                    resp = resp - np.mean(resp[:249])

                all_resp.append(resp)

            all_resp = np.array(all_resp)

            im = ax[il][idur].imshow(
                all_resp, origin='lower', interpolation='none', aspect='auto'
            )
            min_clim_val = -5
            max_clim_val = 50
            im.set_clim([min_clim_val, max_clim_val])

    fig.savefig(
        os.path.join(
            nwb_validation_dir, 'optotagging', str(sessionID) + '.png'
        )
    )
    plt.close('all')


if __name__ == '__main__':
    nwb_base = r'\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\nwbs_220429'
    nwb_paths = glob.glob(os.path.join(nwb_base, '*nwb'))
    pool = Pool()
    pool.map(opto_validation, nwb_paths)

import os

import numpy as np
import pandas as pd
from allensdk.brain_observatory.ecephys.align_timestamps.barcode import (
    extract_barcodes_from_times,
    get_probe_time_offset,
)
from allensdk.brain_observatory.sync_dataset import Dataset as SyncDataset
from matplotlib import pyplot as plt

# GET DATA
syncDataset = SyncDataset(
    r'\\allen\programs\mindscope\workgroups\np-exp\VBN_timing_validation\20220516T183524.h5'
)
datPath = r'\\allen\programs\mindscope\workgroups\np-exp\VBN_timing_validation\2022-05-16_18-35-30\Record Node 105\experiment1\recording1\continuous\NI-DAQmx-103.0\continuous.dat'
ttlStatesPath = r'\\allen\programs\mindscope\workgroups\np-exp\VBN_timing_validation\2022-05-16_18-35-30\Record Node 105\experiment1\recording1\events\NI-DAQmx-103.0\TTL_1\channel_states.npy'
ttlTimestampsPath = os.path.join(
    os.path.dirname(ttlStatesPath), 'timestamps.npy'
)
datTimestampsPath = os.path.join(os.path.dirname(datPath), 'timestamps.npy')
stim_table = pd.read_csv(
    r'\\allen\programs\mindscope\workgroups\np-exp\VBN_timing_validation\stim_table\stim_table_220517_1204.csv'
)

diodeChannels = [0, 2]   # analog channels for diode signals
ephysSampleRate = 30000
numAnalogCh = 8
datData = np.memmap(datPath, dtype='int16', mode='r')
datData = np.reshape(datData, (int(datData.size / numAnalogCh), -1)).T
diodeData = {k: datData[k] for k in diodeChannels}

# ALIGN EPHYS DATA TO SYNC
get_edges = lambda key: [
    syncDataset.get_rising_edges(key, units='seconds'),
    syncDataset.get_falling_edges(key, units='seconds'),
]
syncBarcodeRising, syncBarcodeFalling = get_edges('barcode_ephys')
syncBarcodeTimes, syncBarcodes = extract_barcodes_from_times(
    syncBarcodeRising, syncBarcodeFalling
)

datTimestamps = np.load(datTimestampsPath)
fullTimestamps = datTimestamps - datTimestamps[0]

ttlStates = np.load(ttlStatesPath)
ttlTimestamps = np.load(ttlTimestampsPath) - datTimestamps[0]

ephysBarcodeRising = ttlTimestamps[ttlStates > 0] / ephysSampleRate
ephysBarcodeFalling = ttlTimestamps[ttlStates < 0] / ephysSampleRate
ephysBarcodeTimes, ephysBarcodes = extract_barcodes_from_times(
    ephysBarcodeRising, ephysBarcodeFalling
)

ephysShift, relSampleRate, endpoints = get_probe_time_offset(
    syncBarcodeTimes,
    syncBarcodes,
    ephysBarcodeTimes,
    ephysBarcodes,
    0,
    ephysSampleRate,
)
fullTimestamps = (fullTimestamps / relSampleRate) - ephysShift


# PLOT DIODE SIGNAL ALIGNED TO TIMESTAMPS
behavior_stim_table = stim_table.loc[stim_table['active']]
mapping_stim_table = stim_table.loc[stim_table['color'] == -1]
passive_stim_table = stim_table.loc[stim_table['stimulus_block'] == 5]


def mean_aligned(stim_times, diodeData):
    aligned_traces = [[] for d in diodeData]
    for stimtime in stim_times:

        start_sample_ind = np.searchsorted(fullTimestamps, stimtime)
        mean_start = start_sample_ind - int(relSampleRate)
        mean_end = start_sample_ind + int(relSampleRate)

        for ichan, chan in enumerate(diodeData):
            aligned_traces[ichan].append(diodeData[chan][mean_start:mean_end])

    means = [np.mean(dt, axis=0) for dt in aligned_traces]
    means = [
        mean - np.mean(mean[: int(0.8 * relSampleRate)]) for mean in means
    ]
    time = np.linspace(-1, 1, len(means[0]))
    return means, time


def plot_diode(stim_times, diodeData, ax, legend=None, title=''):

    means, time = mean_aligned(stim_times, diodeData)
    [ax.plot(time, mean) for mean in means]
    ax.set_xlim([-0.02, 0.05])
    ax.axvline(0, linestyle='--')
    ax.set_xlabel('Time from stim onset (s)')
    ax.set_title(title)
    if legend:
        ax.legend(legend)


fig, axes = plt.subplots(1, 3)
fig.set_size_inches([14, 6])
plot_diode(
    behavior_stim_table.start_time.values,
    diodeData,
    axes[0],
    None,
    'behavior stim',
)
plot_diode(
    mapping_stim_table.start_time.values,
    diodeData,
    axes[1],
    None,
    'mapping stim',
)
plot_diode(
    passive_stim_table.start_time.values,
    diodeData,
    axes[2],
    ['center', 'top'],
    'passive_stim',
)


# PLOT DIODE 'RECEPTIVE FIELDS'
rf_stim_table = stim_table.loc[stim_table['stimulus_block'] == 2]
xs = np.sort(rf_stim_table.position_x.unique())
ys = np.sort(rf_stim_table.position_y.unique())
fig, ax = plt.subplots(len(ys), len(xs))
for ix, x in enumerate(xs):
    for iy, y in enumerate(ys):
        rfdf = rf_stim_table.loc[
            (rf_stim_table['position_x'] == x)
            & (rf_stim_table['position_y'] == y)
        ]
        means, time = mean_aligned(rfdf.start_time.values, diodeData)
        [ax[ys.size - iy - 1][ix].plot(time[::2], mean[::2]) for mean in means]

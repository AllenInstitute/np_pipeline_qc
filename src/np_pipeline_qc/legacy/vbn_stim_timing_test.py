# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:49:18 2022

@author: svc_ccg
"""
import os

import numpy as np
import pandas as pd
from allensdk.brain_observatory.ecephys.file_io.stim_file import (
    CamStimOnePickleStimFile,
)
from matplotlib import pyplot as plt

import np_pipeline_qc.legacy.probeSync_qc as probeSync
from np_pipeline_qc.legacy import build_stim_tables, ecephys
from np_pipeline_qc.legacy.sync_dataset import Dataset

behavior_data = pd.read_pickle(
    r'C:\Users\svc_ccg\Desktop\VBN_timing_test\220510180649.pkl'
)
mapping_data = pd.read_pickle(
    r'C:\Users\svc_ccg\Desktop\VBN_timing_test\220510183227-script.pkl'
)

syncDataset = Dataset(
    r'C:\Users\svc_ccg\Desktop\VBN_timing_test\20220510T175349.h5'
)

datPath = r'C:\Users\svc_ccg\Desktop\VBN_timing_test\2022-05-10_17-54-24\Record Node 105\experiment1\recording1\continuous\NI-DAQmx-103.0\continuous.dat'
ttlStatesPath = r'C:\Users\svc_ccg\Desktop\VBN_timing_test\2022-05-10_17-54-24\Record Node 105\experiment1\recording1\events\NI-DAQmx-103.0\TTL_1\channel_states.npy'
ttlTimestampsPath = os.path.join(
    os.path.dirname(ttlStatesPath), 'timestamps.npy'
)
datTimestampsPath = os.path.join(os.path.dirname(datPath), 'timestamps.npy')


vsyncRising, vsyncFalling = probeSync.get_sync_line_data(
    syncDataset, 'vsync_stim'
)
vsyncTimes = (
    vsyncFalling[1:] if vsyncFalling[0] < vsyncRising[0] else vsyncFalling
)

syncBarcodeRising, syncBarcodeFalling = probeSync.get_sync_line_data(
    syncDataset, 'barcode_ephys'
)
syncBarcodeTimes, syncBarcodes = ecephys.extract_barcodes_from_times(
    syncBarcodeRising, syncBarcodeFalling
)

ephysSampleRate = 30000

numAnalogCh = 8
datData = np.memmap(datPath, dtype='int16', mode='r')
datData = np.reshape(datData, (int(datData.size / numAnalogCh), -1)).T

datTimestamps = np.load(datTimestampsPath)
fullTimestamps = datTimestamps - datTimestamps[0]

diodeCh = 0
diodeData = datData[diodeCh]

ttlStates = np.load(ttlStatesPath)
ttlTimestamps = np.load(ttlTimestampsPath) - datTimestamps[0]

ephysBarcodeRising = ttlTimestamps[ttlStates > 0] / ephysSampleRate
ephysBarcodeFalling = ttlTimestamps[ttlStates < 0] / ephysSampleRate
ephysBarcodeTimes, ephysBarcodes = ecephys.extract_barcodes_from_times(
    ephysBarcodeRising, ephysBarcodeFalling
)

ephysShift, relSampleRate, endpoints = ecephys.get_probe_time_offset(
    syncBarcodeTimes,
    syncBarcodes,
    ephysBarcodeTimes,
    ephysBarcodes,
    0,
    ephysSampleRate,
)

fullTimestamps = (fullTimestamps / relSampleRate) - ephysShift


behavior_stim_table = build_stim_tables.generate_behavior_stim_table(
    behavior_data, syncDataset
)

stim_times = behavior_stim_table.Start.values
diode_traces = []
for stimtime in stim_times:

    start_sample_ind = np.searchsorted(fullTimestamps, stimtime)
    mean_start = start_sample_ind - int(relSampleRate)
    mean_end = start_sample_ind + int(relSampleRate)
    diode_traces.append(diodeData[mean_start:mean_end])


mean = np.mean(diode_traces, axis=0)
time = np.linspace(-1, 1, len(mean))
fig, ax = plt.subplots()
fig.suptitle('behavior image stim timing')
ax.plot(time, mean)
ax.set_xlim([-0.02, 0.05])
ax.axvline(0, linestyle='--')
ax.set_xlabel('Time from stim onset (s)')

mapping_data_stimuli = CamStimOnePickleStimFile(mapping_data)
mapping_stim_table = build_stim_tables.generate_mapping_stim_table(
    mapping_data_stimuli,
    syncDataset,
    1,
    behavior_data['items']['behavior']['intervalsms'].size + 1,
)

dark_flashes = mapping_stim_table.loc[mapping_stim_table['Color'] == -1]
flash_times = dark_flashes['Start'].values
diode_traces = []
for stimtime in flash_times:

    start_sample_ind = np.searchsorted(fullTimestamps, stimtime)
    mean_start = start_sample_ind - int(relSampleRate)
    mean_end = start_sample_ind + int(relSampleRate)
    diode_traces.append(diodeData[mean_start:mean_end])


mean = np.mean(diode_traces, axis=0)
time = np.linspace(-1, 1, len(mean))
fig, ax = plt.subplots()
fig.suptitle('mapping flash stim timing')
ax.plot(time, mean)
ax.set_xlim([-0.02, 0.05])
ax.axvline(0, linestyle='--')
ax.set_xlabel('Time from stim onset (s)')

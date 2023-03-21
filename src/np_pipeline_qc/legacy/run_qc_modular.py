# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:42:31 2020

@author: svc_ccg
"""

import glob
import logging
import os
import shutil

import cv2
import numpy as np
import pandas as pd
import scipy.signal
from matplotlib import pyplot as plt

import np_pipeline_qc.legacy.probeSync_qc as probeSync
from np_pipeline_qc.legacy import analysis, behavior_analysis, data_getters
from np_pipeline_qc.legacy.get_RFs_standalone import get_RFs
#from visual_behavior.ophys.sync import sync_dataset
from np_pipeline_qc.legacy.sync_dataset import Dataset as sync_dataset

### SPECIFY EXPERIMENT TO PULL ####
#this should be either the ten digit lims id:
#identifier = '1013651431' #lims id
#or the local base directory
identifier = r'\\10.128.50.43\sd6.3\1033616558_509940_20200701'
identifier = r'\\10.128.50.43\sd6.3\1033388795_509652_20200630' 
identifier = r"\\10.128.50.43\sd6.3\1033387557_509940_20200630"
identifier = r"\\10.128.50.43\sd6.3\1037927382_513573_20200722"
identifier = r"\\10.128.50.43\sd6.3\1038127711_513573_20200723"

identifier = '1041287144'
if identifier.find('_')>=0:
    d = data_getters.local_data_getter(base_dir=identifier)
else:
    d = data_getters.lims_data_getter(exp_id=identifier)

paths = d.data_dict
FIG_SAVE_DIR = os.path.join(r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\QC", 
                            paths['es_id']+'_'+paths['external_specimen_name']+'_'+paths['datestring'])
if not os.path.exists(FIG_SAVE_DIR):
    os.mkdir(FIG_SAVE_DIR)

### GET FILE PATHS TO SYNC AND PKL FILES ###
SYNC_FILE = paths['sync_file']
BEHAVIOR_PKL = paths['behavior_pkl']
REPLAY_PKL = paths['replay_pkl']
MAPPING_PKL = paths['mapping_pkl']

for f,s in zip([SYNC_FILE, BEHAVIOR_PKL, REPLAY_PKL, MAPPING_PKL], ['sync: ', 'behavior: ', 'replay: ', 'mapping: ']):
    print(s + f)
    

### GET MAIN DATA STREAMS ###
syncDataset = sync_dataset(SYNC_FILE)
behavior_data = pd.read_pickle(BEHAVIOR_PKL)
mapping_data = pd.read_pickle(MAPPING_PKL)
replay_data = pd.read_pickle(REPLAY_PKL)

### Behavior Analysis ###
trials = behavior_analysis.get_trials_df(behavior_data)
behavior_analysis.plot_behavior(trials, FIG_SAVE_DIR)

trial_types, counts = behavior_analysis.get_trial_counts(trials)
behavior_analysis.plot_trial_type_pie(counts, trial_types, FIG_SAVE_DIR)


### CHECK FRAME COUNTS ###
vr, vf = probeSync.get_sync_line_data(syncDataset, channel=2)


    
behavior_frame_count = behavior_data['items']['behavior']['intervalsms'].size + 1
mapping_frame_count = mapping_data['intervalsms'].size + 1
replay_frame_count = replay_data['intervalsms'].size + 1

total_pkl_frames = (behavior_frame_count +
                    mapping_frame_count +
                    replay_frame_count) 

### CHECK THAT NO FRAMES WERE DROPPED FROM SYNC ###
print('frames in pkl files: {}'.format(total_pkl_frames))
print('frames in sync file: {}'.format(len(vf)))

assert(total_pkl_frames==len(vf))

### CHECK THAT REPLAY AND BEHAVIOR HAVE SAME FRAME COUNT ###
print('frames in behavior stim: {}'.format(behavior_frame_count))
print('frames in replay stim: {}'.format(replay_frame_count))

assert(behavior_frame_count==replay_frame_count)

# look for potential frame offsets from aborted stims
(behavior_start_frame, mapping_start_frame, replay_start_frame) = probeSync.get_frame_offsets(syncDataset, 
                                                        [behavior_frame_count,
                                                         mapping_frame_count,
                                                         replay_frame_count])

MONITOR_LAG = 0.036 #TO DO: don't hardcode this...
FRAME_APPEAR_TIMES = vf + MONITOR_LAG  

analysis.plot_frame_intervals(vf, behavior_frame_count, mapping_frame_count, 
                              behavior_start_frame, mapping_start_frame,
                              replay_start_frame, FIG_SAVE_DIR) 


### BUILD UNIT TABLE ####
probe_dict = probeSync.build_unit_table(paths['data_probes'], paths, syncDataset)

analysis.plot_unit_quality_hist(probe_dict, FIG_SAVE_DIR)
analysis.plot_unit_distribution_along_probe(probe_dict, FIG_SAVE_DIR)

get_RFs(probe_dict, mapping_data, mapping_start_frame, FRAME_APPEAR_TIMES, FIG_SAVE_DIR)

analysis.plot_population_change_response(probe_dict, behavior_frame_count, mapping_frame_count, trials, 
                                         FRAME_APPEAR_TIMES, FIG_SAVE_DIR, ctx_units_percentile=66)


analysis.plot_running_wheel(behavior_data, mapping_data, replay_data, FIG_SAVE_DIR)
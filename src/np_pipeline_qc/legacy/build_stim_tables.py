# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:50:27 2020

@author: svc_ccg
"""

import logging
from functools import partial

import numpy as np
import pandas as pd
from allensdk.brain_observatory.behavior.stimulus_processing import \
    get_stimulus_presentations
from allensdk.brain_observatory.ecephys.file_io.stim_file import \
    CamStimOnePickleStimFile
from allensdk.brain_observatory.ecephys.stimulus_table.ephys_pre_spikes import (
    build_stimuluswise_table, create_stim_table,
    make_spontaneous_activity_tables)
from allensdk.brain_observatory.sync_dataset import Dataset

import np_pipeline_qc.legacy.probeSync_qc as probeSync


def get_frame_offsets(sync_dataset, frame_counts, tolerance=0):
    ''' Tries to infer which vsyncs correspond to the frames in the epochs in frame_counts
        This allows you to align data even when there are aborted stimuli
        
        INPUTS:
            sync_dataset: sync data from experiment (a 'Dataset' object made from the H5 file)
            
            frame_counts: list of the expected frame counts (taken from pkl files) for each
                        of the stimuli in question;
                        the list should be ordered by the display sequence
            
            tolerance: percent by which frame counts are allowed to deviate from expected
                        
        OUTPUTS:
            start_frames: list of the inferred start frames for each of the stimuli
    '''
    
    frame_counts = np.array(frame_counts)
    tolerance = tolerance/100.
    
    # get vsyncs and stim_running signals from sync
    #vf = get_vsyncs(sync_dataset)
    vf = probeSync.get_experiment_frame_times(sync_dataset)
    stimstarts, stimoffs = get_stim_starts_ends(sync_dataset)
    
    # get vsync frame lengths for all stimuli
    epoch_frame_counts = []
    epoch_start_frames = []
    for start, end in zip(stimstarts, stimoffs):
        epoch_frames = np.where((vf>start)&(vf<end))[0]
        epoch_frame_counts.append(len(epoch_frames))
        epoch_start_frames.append(epoch_frames[0])
        
    if len(epoch_frame_counts)>len(frame_counts):
        logging.warning('Found extra stim presentations. Inferring start frames')
        
        start_frames = []
        for stim_num, fc in enumerate(frame_counts):
            
            print('finding stim start for stim {}'.format(stim_num))
            best_match = np.argmin([np.abs(e-fc) for e in epoch_frame_counts])
            if fc*(1-tolerance) <= epoch_frame_counts[best_match] <= fc*(1+tolerance):
                _ = epoch_frame_counts.pop(best_match)
                start_frame = epoch_start_frames.pop(best_match)
                start_frames.append(start_frame)
                print('found stim start at vsync {}'.format(start_frame))
                
            else:
                logging.error('Could not find matching sync frames for stim {}'.format(stim_num))
                return
    
    else:        
        start_frames = epoch_start_frames
    
    return start_frames



def generate_behavior_stim_table(pkl_data, sync_dataset, frame_offset=0, block_offset=0):
    
    p = pkl_data
    image_set =  p['items']['behavior']['params']['stimulus']['params']['image_set']
    image_set = image_set.split('/')[-1].split('.')[0]
    num_frames = p['items']['behavior']['intervalsms'].size + 1
    reward_frames = p['items']['behavior']['rewards'][0]['reward_times'][:, 1]
    
    frame_timestamps = probeSync.get_experiment_frame_times(sync_dataset)#get_vsyncs(sync_dataset)
    reward_times = frame_timestamps[reward_frames.astype(int)]
    epoch_timestamps = frame_timestamps[frame_offset:frame_offset+num_frames]
    
    stim_table = get_stimulus_presentations(p, epoch_timestamps)
    stim_table['stimulus_block'] = block_offset
    stim_table['stimulus_name'] = image_set
    stim_table = stim_table.rename(columns={'frame':'start_frame', 'start_time':'Start', 'stop_time':'End'})
    
    # add columns for change and flashes since change
    change = np.zeros(len(stim_table))
    rewarded = np.zeros(len(stim_table))
    repeat_number = np.zeros(len(stim_table))
    current_image = stim_table.iloc[0]['stimulus_name']
    for index, row in stim_table.iterrows():
        if (row['image_name'] == 'omitted') or (row['omitted']) :
            repeat_number[index] = repeat_number[index-1]
        else:
            if row['image_name'] != current_image:
                change[index] = 1
                repeat_number[index] = 0
                current_image = row['image_name']
                if np.min(np.abs(row['Start'] - reward_times))<1:
                    rewarded[index] = 1
            else:
                repeat_number[index] = repeat_number[index-1] + 1
    
    #don't call first change a change
    change[np.where(change)[0][0]] = 0
    stim_table['change'] = change.astype(int)
    stim_table['rewarded'] = rewarded.astype(int)
    
    #stim_table.loc[0, 'change'] = 0 # MAKE BETTER SOMETIMES THIS IS THE SECOND ROW SINCE THE FIRST CAN BE OMITTED
    stim_table['flashes_since_change'] = repeat_number.astype(int)
    stim_table['active'] = True    
    

    # Fill in 'end frame' and 'End' for omitted stimuli
    median_stim_frame_duration = np.nanmedian(stim_table['end_frame'] - stim_table['start_frame'])
    stim_table.loc[stim_table['omitted'], 'end_frame'] = stim_table[stim_table['omitted']]['start_frame'] + median_stim_frame_duration
    stim_table.loc[stim_table['omitted'], 'End'] = epoch_timestamps[stim_table[stim_table['omitted']]['end_frame'].astype(int)]    
    stim_table['common_name'] = 'behavior'
    
    
    return stim_table

    
def generate_replay_stim_table(pkl_data, sync_dataset, behavior_stim_table, block_offset=3, frame_offset=0):
    
    p = pkl_data
    num_frames = p['intervalsms'].size + 1
    
    
    frame_timestamps = probeSync.get_experiment_frame_times(sync_dataset)#get_vsyncs(sync_dataset)
    frame_timestamps = frame_timestamps[frame_offset:frame_offset+num_frames]
    
    ims = p['stimuli'][0]['sweep_params']['ReplaceImage'][0]
    im_names = np.unique([img for img in ims if img is not None])
    
    ## CHECK THAT REPLAY MATCHES BEHAVIOR
    im_ons = []
    im_offs = []
    im_names = []
    for ind, im in enumerate(ims):
        if ind==0:
            continue
        elif ind<len(ims)-1:
            if ims[ind-1] is None and ims[ind] is not None:
                im_ons.append(ind)
                im_names.append(im)
            elif ims[ind] is not None and ims[ind+1] is None:
                im_offs.append(ind)
     
    inter_flash_interval = np.diff(im_ons)
    putative_omitted = np.where(inter_flash_interval>70)[0]
    im_names_with_omitted = np.insert(im_names, putative_omitted+1, 'omitted')
    
    ### Handle omitted flash edge cases ###
    #check if the first flash was omitted
    first_flash_omitted = behavior_stim_table['image_name'].iloc[0] == 'omitted'
    if first_flash_omitted:
        im_names_with_omitted = np.insert(im_names_with_omitted, 0, 'omitted')
    
    #check if last flash was omitted
    last_flash_omitted = behavior_stim_table['image_name'].iloc[-1] == 'omitted'
    if last_flash_omitted:
        im_names_with_omitted = np.insert(im_names_with_omitted, len(im_names_with_omitted), 'omitted')
    
    # Verify that the image list for replay is identical to behavior
    assert all(behavior_stim_table['image_name'] == im_names_with_omitted)
    
    ## IF SO, JUST USE THE BEHAVIOR STIM TABLE, BUT ADJUST TIMES/FRAMES
    stim_table = behavior_stim_table.copy(deep=True)
    stim_table['stimulus_block'] = block_offset
    stim_table['Start'] = frame_timestamps[stim_table['start_frame']]
    stim_table.loc[:, 'End'] = frame_timestamps[stim_table['end_frame'].dropna().astype(int)]
    stim_table['start_frame'] = stim_table['start_frame'] + frame_offset
    stim_table.loc[:, 'end_frame'] = stim_table['end_frame'] + frame_offset
    stim_table['active'] = False
    stim_table['common_name'] = 'replay'

    return stim_table

def generate_mapping_stim_table(pkl_data, sync_dataset, block_offset=1, frame_offset=0):
    
    stim_file = pkl_data
    
    seconds_to_frames = (
        lambda seconds: (np.array(seconds) + stim_file.pre_blank_sec)
        * stim_file.frames_per_second
    )
    
    stim_tabler = partial(build_stimuluswise_table, seconds_to_frames = seconds_to_frames)
    stim_table = create_stim_table(stim_file.stimuli, stim_tabler, make_spontaneous_activity_tables)
    
    
    frame_timestamps = probeSync.get_experiment_frame_times(sync_dataset)#get_vsyncs(sync_dataset)
    
    stim_table = stim_table.rename(columns={'Start':'start_frame', 'End':'end_frame'})
    stim_table['start_frame'] = np.array(stim_table['start_frame']).astype(int) + frame_offset
    stim_table['end_frame'] = np.array(stim_table['end_frame']).astype(int) + frame_offset
    stim_table['Start'] = frame_timestamps[stim_table['start_frame']]
    stim_table['End'] = frame_timestamps[stim_table['end_frame']]
    stim_table['stimulus_block'] = stim_table['stimulus_block'] + block_offset
    stim_table['active'] = False
    stim_table['common_name'] = 'mapping'

    return stim_table
    
    
def get_vsyncs(sync_dataset, fallback_line=2):
    
    lines = sync_dataset.line_labels
    
    #look for vsyncs in labels
    vsync_line = fallback_line
    for line in lines:
        if 'vsync' in line:
            vsync_line = line
    
    falling_edges = sync_dataset.get_falling_edges(vsync_line, units='seconds')
    
    return falling_edges

        
def get_stim_starts_ends(sync_dataset, fallback_line=5):
    
    lines = sync_dataset.line_labels
    
    #look for vsyncs in labels
    if 'stim_running' in lines:
        stim_line = 'stim_running'
    else:
        stim_line = fallback_line
    
    stim_ons = sync_dataset.get_rising_edges(stim_line, units='seconds')
    stim_offs = sync_dataset.get_falling_edges(stim_line, units='seconds')
    
    return stim_ons, stim_offs
        
        
def sort_columns(dataframe, ordered_cols):
    '''Rearrage columns in dataframe.
        INPUT:
            dataframe: the dataframe you want to resort
            ordered_cols: order of the columns you want.
                You can specify a subset of the total columns here,
                and the function will just tack on the rest ordered
                alphanumerically
    '''
    all_cols = dataframe.columns.tolist()
    
    #if there are more columns than specified in
    #ordered_cols, just tack them on alphabetically
    if len(ordered_cols) < len(all_cols):
        sorted_cols = [c for c in np.sort(all_cols) if c not in ordered_cols]
        final_cols = ordered_cols + sorted_cols
    else:
        final_cols = ordered_cols
    
    return dataframe[final_cols]
        

def build_full_NP_behavior_stim_table(behavior_pkl_path, mapping_pkl_path, replay_pkl_path, sync_path):
    
    pkl_files = []
    for pkl in [behavior_pkl_path, mapping_pkl_path, replay_pkl_path]:
        if isinstance(pkl, str):
            pkl = pd.read_pickle(pkl)
        pkl_files.append(pkl)
    
    behavior_pkl = pkl_files[0]
    mapping_pkl = pkl_files[1]
    replay_pkl = pkl_files[2]
    
    if isinstance(sync_path, str):
        sync_dataset = Dataset(sync_path)
    else:
        sync_dataset = sync_path
        
#    behavior_pkl = pd.read_pickle(behavior_pkl_path)
#    mapping_pkl = pd.read_pickle(mapping_pkl_path)
#    replay_pkl = pd.read_pickle(replay_pkl_path)
    
    frame_counts = []
    for p in [behavior_pkl, mapping_pkl, replay_pkl]:
        total_frames = len(p['intervalsms']) + 1 if 'intervalsms' in p \
                else len(p['items']['behavior']['intervalsms']) + 1
        frame_counts.append(total_frames)
    
    #mapping_stim_file = CamStimOnePickleStimFile.factory(mapping_pkl_path)
    mapping_stim_file = CamStimOnePickleStimFile(mapping_pkl)
    
    frame_offsets = get_frame_offsets(sync_dataset, frame_counts)        
    
    stim_table_behavior = generate_behavior_stim_table(behavior_pkl, sync_dataset, frame_offset=frame_offsets[0])
    stim_table_mapping = generate_mapping_stim_table(mapping_stim_file, sync_dataset, frame_offset=frame_offsets[1])
    stim_table_replay = generate_replay_stim_table(replay_pkl, sync_dataset, stim_table_behavior, frame_offset=frame_offsets[2])
    
    #Rearrange columns to make a bit more readable; the rest of the cols are just alphabetical
    stim_table_behavior = sort_columns(stim_table_behavior, ['stimulus_block',
                                                             'active',
                                                             'stimulus_name',
                                                             'Start',
                                                             'End',
                                                             'duration',
                                                             'start_frame',
                                                             'end_frame'])
    
    stim_table_full = pd.concat([stim_table_behavior, stim_table_mapping, stim_table_replay], sort=False)
    stim_table_full.loc[:, 'duration'] = stim_table_full['End'] - stim_table_full['Start']
    stim_table_full.loc[stim_table_full['stimulus_name'].isnull(), 'stimulus_name'] = 'spontaneous'
    stim_table_full['presentation_index'] = np.arange(len(stim_table_full))
    
    return stim_table_full.set_index('presentation_index')
    
    
    
def get_opto_stim_table(syncDataset, opto_pkl, opto_sample_rate=10000):
    
    trial_levels = opto_pkl['opto_levels']
    trial_conds = opto_pkl['opto_conditions']
    trial_start_times = syncDataset.get_rising_edges('stim_trial_opto', units='seconds')
    
    waveforms = opto_pkl['opto_waveforms']
    trial_waveform_durations = [waveforms[cond].size/opto_sample_rate for cond in trial_conds]
    
    trial_end_times = trial_start_times + trial_waveform_durations
    
    trial_dict = {
            'trial_levels': trial_levels,
            'trial_conditions': trial_conds,
            'trial_start_times': trial_start_times,
            'trial_end_times': trial_end_times}
    
    return trial_dict
    
    
    
    
    
    
    
    
    
    
    
    
    
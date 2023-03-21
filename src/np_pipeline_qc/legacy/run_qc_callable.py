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

# from visual_behavior.ophys.sync import sync_dataset
from np_pipeline_qc.legacy.sync_dataset import Dataset as sync_dataset


def run_qc(exp_id, save_root):

    identifier = exp_id
    if identifier.find('_') >= 0:
        d = data_getters.local_data_getter(base_dir=identifier)
    else:
        d = data_getters.lims_data_getter(exp_id=identifier)

    paths = d.data_dict
    FIG_SAVE_DIR = os.path.join(
        save_root,
        paths['es_id']
        + '_'
        + paths['external_specimen_name']
        + '_'
        + paths['datestring'],
    )
    if not os.path.exists(FIG_SAVE_DIR):
        os.mkdir(FIG_SAVE_DIR)

    figure_prefix = (
        paths['external_specimen_name'] + '_' + paths['datestring'] + '_'
    )

    ### GET FILE PATHS TO SYNC AND PKL FILES ###
    SYNC_FILE = paths['sync_file']
    BEHAVIOR_PKL = paths['behavior_pkl']
    REPLAY_PKL = paths['replay_pkl']
    MAPPING_PKL = paths['mapping_pkl']

    for f, s in zip(
        [SYNC_FILE, BEHAVIOR_PKL, REPLAY_PKL, MAPPING_PKL],
        ['sync: ', 'behavior: ', 'replay: ', 'mapping: '],
    ):
        print(s + f)

    ### GET MAIN DATA STREAMS ###
    syncDataset = sync_dataset(SYNC_FILE)
    behavior_data = pd.read_pickle(BEHAVIOR_PKL)
    mapping_data = pd.read_pickle(MAPPING_PKL)
    replay_data = pd.read_pickle(REPLAY_PKL)

    ### Behavior Analysis ###
    behavior_plot_dir = os.path.join(FIG_SAVE_DIR, 'behavior')
    trials = behavior_analysis.get_trials_df(behavior_data)
    behavior_analysis.plot_behavior(
        trials, behavior_plot_dir, prefix=figure_prefix
    )

    trial_types, counts = behavior_analysis.get_trial_counts(trials)
    behavior_analysis.plot_trial_type_pie(
        counts, trial_types, behavior_plot_dir, prefix=figure_prefix
    )

    ### CHECK FRAME COUNTS ###
    vr, vf = probeSync.get_sync_line_data(syncDataset, channel=2)

    behavior_frame_count = (
        behavior_data['items']['behavior']['intervalsms'].size + 1
    )
    mapping_frame_count = mapping_data['intervalsms'].size + 1
    replay_frame_count = replay_data['intervalsms'].size + 1

    total_pkl_frames = (
        behavior_frame_count + mapping_frame_count + replay_frame_count
    )

    ### CHECK THAT NO FRAMES WERE DROPPED FROM SYNC ###
    print('frames in pkl files: {}'.format(total_pkl_frames))
    print('frames in sync file: {}'.format(len(vf)))

    # assert(total_pkl_frames==len(vf))

    ### CHECK THAT REPLAY AND BEHAVIOR HAVE SAME FRAME COUNT ###
    print('frames in behavior stim: {}'.format(behavior_frame_count))
    print('frames in replay stim: {}'.format(replay_frame_count))

    # assert(behavior_frame_count==replay_frame_count)

    # look for potential frame offsets from aborted stims
    (
        behavior_start_frame,
        mapping_start_frame,
        replay_start_frame,
    ) = probeSync.get_frame_offsets(
        syncDataset,
        [behavior_frame_count, mapping_frame_count, replay_frame_count],
    )

    behavior_end_frame = behavior_start_frame + behavior_frame_count - 1
    mapping_end_frame = mapping_start_frame + mapping_frame_count - 1
    replay_end_frame = replay_start_frame + replay_frame_count - 1

    MONITOR_LAG = 0.036   # TO DO: don't hardcode this...
    FRAME_APPEAR_TIMES = vf + MONITOR_LAG

    behavior_start_time, mapping_start_time, replay_start_time = [
        FRAME_APPEAR_TIMES[f]
        for f in [
            behavior_start_frame,
            mapping_start_frame,
            replay_start_frame,
        ]
    ]
    behavior_end_time, mapping_end_time, replay_end_time = [
        FRAME_APPEAR_TIMES[f]
        for f in [behavior_end_frame, mapping_end_frame, replay_end_frame]
    ]

    ### Plot vsync info ###
    vsync_save_dir = os.path.join(FIG_SAVE_DIR, 'vsyncs')
    analysis.plot_frame_intervals(
        vf,
        behavior_frame_count,
        mapping_frame_count,
        behavior_start_frame,
        mapping_start_frame,
        replay_start_frame,
        vsync_save_dir,
        prefix=figure_prefix,
    )
    analysis.plot_vsync_interval_histogram(
        vf, vsync_save_dir, prefix=figure_prefix
    )
    analysis.vsync_report(
        vf, total_pkl_frames, vsync_save_dir, prefix=figure_prefix
    )

    ### BUILD UNIT TABLE ####
    probe_dict = probeSync.build_unit_table(
        paths['data_probes'], paths, syncDataset
    )

    ### Plot Probe Yield QC ###
    probe_yield_dir = os.path.join(FIG_SAVE_DIR, 'probe_yield')
    probe_dirs = [paths['probe' + pid] for pid in paths['data_probes']]
    analysis.plot_unit_quality_hist(
        probe_dict, probe_yield_dir, prefix=figure_prefix
    )
    analysis.plot_unit_distribution_along_probe(
        probe_dict, probe_yield_dir, prefix=figure_prefix
    )
    analysis.plot_all_spike_hist(
        probe_dict, probe_yield_dir, prefix=figure_prefix + 'good'
    )
    analysis.copy_probe_depth_images(
        paths, probe_yield_dir, prefix=figure_prefix
    )

    ### Unit Metrics ###
    unit_metrics_dir = os.path.join(FIG_SAVE_DIR, 'unit_metrics')
    analysis.plot_unit_metrics(paths, unit_metrics_dir, prefix=figure_prefix)

    ### Probe/Sync alignment
    probeSyncDir = os.path.join(FIG_SAVE_DIR, 'probeSyncAlignment')
    analysis.plot_barcode_interval_hist(
        probe_dirs, syncDataset, probeSyncDir, prefix=figure_prefix
    )
    analysis.plot_barcode_intervals(
        probe_dirs, syncDataset, probeSyncDir, prefix=figure_prefix
    )
    analysis.probe_sync_report(
        probe_dirs, syncDataset, probeSyncDir, prefix=figure_prefix
    )
    analysis.plot_barcode_matches(
        probe_dirs, syncDataset, probeSyncDir, prefix=figure_prefix
    )

    ### Plot visual responses
    get_RFs(
        probe_dict,
        mapping_data,
        mapping_start_frame,
        FRAME_APPEAR_TIMES,
        os.path.join(FIG_SAVE_DIR, 'receptive_fields'),
        prefix=figure_prefix,
    )
    analysis.plot_population_change_response(
        probe_dict,
        behavior_frame_count,
        mapping_frame_count,
        trials,
        FRAME_APPEAR_TIMES,
        os.path.join(FIG_SAVE_DIR, 'change_response'),
        ctx_units_percentile=66,
        prefix=figure_prefix,
    )

    ### Plot running ###
    analysis.plot_running_wheel(
        behavior_data,
        mapping_data,
        replay_data,
        behavior_plot_dir,
        prefix=figure_prefix,
    )

    ### LFP ###
    lfp_save_dir = os.path.join(FIG_SAVE_DIR, 'LFP')
    lick_times = analysis.get_rewarded_lick_times(
        probeSync.get_lick_times(syncDataset),
        FRAME_APPEAR_TIMES,
        trials,
        min_inter_lick_time=0.5,
    )
    lfp_dict = probeSync.build_lfp_dict(probe_dirs, syncDataset)
    analysis.plot_lick_triggered_LFP(
        lfp_dict,
        lick_times,
        lfp_save_dir,
        prefix=figure_prefix,
        agarChRange=None,
        num_licks=20,
        windowBefore=0.5,
        windowAfter=1.5,
        min_inter_lick_time=0.5,
        behavior_duration=3600,
    )

    ### VIDEOS ###
    video_dir = os.path.join(FIG_SAVE_DIR, 'videos')
    analysis.lost_camera_frame_report(paths, video_dir, prefix=figure_prefix)
    analysis.camera_frame_grabs(
        paths,
        syncDataset,
        video_dir,
        [behavior_start_time, mapping_start_time, replay_start_time],
        [behavior_end_time, mapping_end_time, replay_end_time],
        epoch_frame_nums=[2, 2, 2],
        prefix=figure_prefix,
    )

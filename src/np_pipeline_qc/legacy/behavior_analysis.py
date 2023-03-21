# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:48:43 2020

@author: svc_ccg
"""

import os

import numpy as np
from matplotlib import pyplot as plt
from visual_behavior.change_detection.trials.session_metrics import \
    trial_count_by_trial_type
from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.visualization.extended_trials.daily import \
    make_daily_figure

#from visual_behavior.ophys.sync import sync_dataset
from np_pipeline_qc.legacy.sync_dataset import Dataset as sync_dataset


def get_trials_df(behavior_data):
    
    core_data = data_to_change_detection_core(behavior_data)
    trials = create_extended_dataframe(
        trials=core_data['trials'],
        metadata=core_data['metadata'],
        licks=core_data['licks'],
        time=core_data['time'])
    
    return(trials)


def plot_behavior(trials, save_dir=None, prefix=''):
    
    daily_behavior_fig = make_daily_figure(trials)
    if save_dir:
        save_figure(daily_behavior_fig, os.path.join(save_dir, prefix+'behavior_summary.png'))
    
    
def get_trial_counts(trials):
    
    trial_counts = []
    labels = []
    for tt in ['hit', 'miss', 'correct_reject', 'false_alarm', 'aborted']:
        trial_counts.append(trial_count_by_trial_type(trials, tt))
        labels.append(tt)
        
    return labels, trial_counts


def plot_trial_type_pie(trial_counts, labels, save_dir=None, prefix=''):
    
    colors = ['g', '0.5', 'b', 'r', 'orange']
    fig, ax = plt.subplots()
    fig.suptitle('trial types')

    def func(pct, allvals):
        absolute = int(round(pct/100.*np.sum(allvals)))
        #return "{:.1f}%\n({:d})".format(pct, absolute)
        return str(absolute)


    wedges, texts, autotexts = ax.pie(trial_counts, colors=colors, 
                                      autopct=lambda pct: func(pct, trial_counts),
                                      textprops=dict(color="w")) 
    ax.legend(wedges, labels,
          title="Trial Types",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
    
    if save_dir:
        save_figure(fig, os.path.join(save_dir, prefix+'trial_type_piechart.png'))


def plot_trial_licks(trials, frame_times, behavior_start_frame, save_dir=None, prefix=''):
    
    frame_times = np.copy(frame_times)
    frame_times = frame_times[behavior_start_frame:]
    
    lick_frames = trials['lick_frames']
    lick_times_trial_binned = [frame_times[f] for f in lick_frames]
    
    #change_frames = np.array(trials['change_frame'].dropna()).astype(int)+1
    change_frames = trials['change_frame']
    resp_types =  ['MISS', 'HIT', 'FA', 'CR']
    colors = ['orange', 'g', 'r', 'b']
    fig, axes = plt.subplots(1, len(resp_types))
    fig.set_size_inches([12, 4])
    for ir, resp_type in enumerate(resp_types):
        ax = axes[ir]
        ax.set_title(resp_type)
        #change_time_line = ax.axvline(0, c='k')
        lick_window_start = ax.axvline(0.15, c='k', linestyle='--')
        lick_window_end = ax.axvline(0.75, c='k', linestyle='--')
        trial_counter = 0
        for t, trial in trials.iterrows():
            if trial['response_type'] == resp_type:
                lts = lick_times_trial_binned[t]
                change_frame = int(change_frames[t])
                change_time = frame_times[change_frame]
                
                lts = lts - change_time
                ax.plot(lts, trial_counter*np.ones(len(lts)), '|', c = colors[ir], ms = 3, markeredgewidth=3)
                trial_counter += 1
        ax.set_xlim([-0.5, 2])
        ax.set_xlabel('Time from change (s)')
        if ir == 0:
            ax.legend([lick_window_start], ['response win'])
            ax.set_ylabel('trials')     
    
    if save_dir is not None:
        save_figure(fig, os.path.join(save_dir, prefix+'trial_licks.png'))

    
def save_figure(fig, save_path):
    
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    fig.savefig(save_path) 
    
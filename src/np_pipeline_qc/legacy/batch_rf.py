# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 09:03:21 2020

@author: svc_ccg
"""

import datetime
import glob
import os
import pickle
import re
import subprocess

import numpy as np
import scipy.ndimage
import scipy.signal
from matplotlib import pyplot as plt

from np_pipeline_qc.legacy.get_sessions import get_sessions

sources = [r"\\10.128.50.43\sd6.3", r"\\10.128.50.20\sd7", r"\\10.128.50.20\sd7.2", r"\\10.128.54.20\sd8"]
sessionsToRun = get_sessions(sources, mouseID='!366122', start_date='20200930')#, end_date='20200922')
destination_folder = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\QC\rf_summary"

#sessionsToRun = get_sessions(source_root, mouseID='!366122', rig='NP1', start_date='20200601')
rf_script = r"C:\Users\svc_ccg\Documents\GitHub\NP_pipeline_QC\get_RFs_standalone.py"
failed = []
for s in sessionsToRun:
    try:
        #check whether npy file already exists for this session
        s_base = os.path.basename(s)
        rf_file = glob.glob(os.path.join(destination_folder, s_base+'*'))
        
        if len(rf_file)==0:
            print('running session {}'.format(s))
            command_string = ['python', rf_script, s, '--save', '--save_dir', destination_folder]
            print(command_string)
            subprocess.check_call(command_string)
        else:
            print('found existing rf file for session {}'.format(s))
    except:
        print('failed to run session {}'.format(s))
        failed.append(s)

data_directory = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\QC\rf_summary'
sessions = os.listdir(data_directory)
sessions = [os.path.join(data_directory,s) for s in sessions if 'npy' in s]

rf_summary = {p:{'peak_chan':[], 'rf_mats':[], 'session':[]} for p in 'ABCDEF'}
for s in sessions:
    try:
        print('loading session {}'.format(s))
    #    with open(s, 'rb') as file:
    #        rf_data = pickle.load(file)
        rf_data = np.load(s, allow_pickle=True)
        if not isinstance(rf_data, dict):
             #weird hack to deal with the fact that some data was pickled 
             #and some was np.saved (hidden pickle). Doing this somehow 
             #rescues the latter data and returns a dict...
            rf_data = rf_data.tolist()
            
        for p in rf_data:
            peak_chan_key = 'peakChan' if 'peakChan' in rf_data[p] else 'peak_channel'
            rf_summary[p]['peak_chan'].append(rf_data[p][peak_chan_key])
            rf_summary[p]['rf_mats'].append(rf_data[p]['rfmat'])
            rf_summary[p]['session'].append(s)
    except:
       print('failed to load {}'.format(s))


def get_rf_center_of_mass(rfmat, exp=5):
    
    if rfmat.ndim>2:
        rfmat = np.mean(rfmat, axis=2)
    
    rfmat = rfmat - rfmat.min()
    rfmat = rfmat**exp
    com = scipy.ndimage.center_of_mass(rfmat)
    
    return com[1], com[0]

def get_rf_max_position(rfmat):
    
    if rfmat.ndim>2:
        rfmat = np.mean(rfmat, axis=2)
        
    max_loc = np.unravel_index(np.argmax(rfmat), rfmat.shape)
    
    return max_loc[1], max_loc[0]

def get_significant_rf(rfmat, nreps=1000, conv=2, sig_percentile=95):
    
    if rfmat.ndim>2:
        rfmat = np.mean(rfmat, axis=2)
    
    conv_mat = np.ones((2,2))
    rf_conv = scipy.signal.convolve2d(rfmat, conv_mat, 'same')/4
    
    shuffled = []
    rf_shuff = np.copy(rfmat)
    for rep in np.arange(nreps):
        flat = rf_shuff.flatten()
        np.random.shuffle(flat)
        unflat = flat.reshape([9,9])
        unflat_conv = scipy.signal.convolve2d(unflat, conv_mat, 'same')/4
        shuffled.append(unflat_conv)
    
    shuff_max = [s.max() for s in shuffled]
    percentile = np.percentile(shuff_max, sig_percentile)
    
    return rf_conv.max() > percentile

def formataxes(ax, title=None, xLabel=None, yLabel=None, 
               xTickLabels=None, yTickLabels=None, no_spines=False,
               ylims=None, xlims=None, spinesToHide=None):
    
    if spinesToHide is None:
        spinesToHide = ['right', 'top', 'left', 'bottom'] if no_spines else ['right', 'top']
    for spines in spinesToHide:
        ax.spines[spines].set_visible(False)

    ax.tick_params(direction='out',top=False,right=False)
    
    if title is not None:
        ax.set_title(title)
    if xLabel is not None:
        ax.set_xlabel(xLabel)
    if yLabel is not None:
        ax.set_ylabel(yLabel)
    if ylims is not None:
        ax.set_ylim(ylims)
    if xlims is not None:
        ax.set_xlim(xlims)   

significant_rfs = {p:[[],[]] for p in 'ABCDEF'}
for p in 'ABCDEF':
    
#    fig, ax = plt.subplots()
#    fig.suptitle(p)
    
    p_rfs = []
    for sess, s in zip(rf_summary[p]['session'], rf_summary[p]['rf_mats']):
        significant = [get_significant_rf(r) for r in s]
        sig_rfs = [r for ir,r in enumerate(s) if significant[ir]]
        p_rfs.extend(sig_rfs)
        significant_rfs[p][0].extend(sig_rfs)
        significant_rfs[p][1].extend([sess]*len(sig_rfs))
    
    
#    #coms = [get_rf_max_position(r) for r in p_rfs]
#    coms =  [get_rf_center_of_mass(r) for r in p_rfs]
#    ax.plot([c[0] for c in coms], [c[1] for c in coms], 'ko', alpha=0.5)
#    ax.set_aspect('equal')
#    ax.set_xlim([0, 8])
#    ax.set_ylim([0, 8])    
area_dict = {'A': 'AM', 'B': 'PM', 'C': 'V1', 'D':'LM', 'E':'AL', 'F':'RL'}
com_exp = 5
fig, axes = plt.subplots(1,6)
fig.set_size_inches([14,6])
for ip, p in enumerate('CDFEBA'):
#    fig, ax = plt.subplots()
#    fig.suptitle(area_dict[p])
    ax = axes[ip]
    ax.set_title(area_dict[p])
    
    p_rfs = significant_rfs[p][0]
    session_dates = [int(sid.split('_')[-1][:8]) for sid in significant_rfs[p][1]]
    
    split_date = 20211225
    coms =  [get_rf_center_of_mass(r, exp=com_exp) for r, sess in zip(p_rfs, session_dates) if sess<split_date]
    ax.plot([c[0] for c in coms], [c[1] for c in coms], 'o', color='k', alpha=0.08, ms=5, markeredgecolor='none')
    
    coms =  [get_rf_center_of_mass(r, exp=com_exp) for r, sess in zip(p_rfs, session_dates) if sess>=split_date]
    ax.plot([c[0] for c in coms], [c[1] for c in coms], 'ro', alpha=0.7)
    
    ax.set_aspect('equal')
    ax.set_xlim([0, 8])
    ax.set_ylim([0, 8])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    #[ax.spines[loc].set_visible(False) for loc in ['top', 'bottom', 'left', 'right']]
    formataxes(ax, no_spines=True)
    
nowstring = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
fig.savefig(os.path.join(data_directory, 'all_rfs_' + nowstring + '.png'))

p_rfs = significant_rfs['A'][0]
for rf in p_rfs[::50]:
    
    fig, ax = plt.subplots()
    ax.imshow(np.mean(rf, axis=2))
    com = get_rf_center_of_mass(rf, exp=com_exp)
    ax.plot(com[0], com[1], 'ro')    
    
    
total_cells = {a:[] for a in 'ABCDEF'}
for p in 'ABCDEF':
    p_rfs = []
    for sess, s in zip(rf_summary[p]['session'], rf_summary[p]['rf_mats']):
        total_cells[p].append(len(s))

fraction_with_rfs = np.array([len(significant_rfs[p][1]) for p in 'ABCDEF'])/np.array([np.sum(total_cells[p]) for p in 'ABCDEF'])







    
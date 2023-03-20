# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:27:14 2022

@author: svc_ccg
"""

import EcephysBehaviorSession as ebs
import os, glob
import pandas as pd
import analysis
import numpy as np

df = pd.read_excel(r"C:\Users\svc_ccg\ccb_onedrive\OneDrive - Allen Institute\all_np_behavior_mice.xlsx")
opt_data_dir = r'\\allen\programs\mindscope\workgroups\np-behavior\processed_ALL'

def vis_mod(spikes, stim_times, monitor_lag=0.02):
    if isinstance(spikes, float):
        vm = 0
    else:
        psth, _ = analysis.makePSTH_numba(spikes.flatten(), stim_times-0.5, 0.75)
        vm = (np.mean(psth[500:]) - np.mean(psth[:500]))/(np.mean(psth[500:]) + np.mean(psth[:500]))
    
    return abs(vm)
    


h5_dir = r"C:\Data\NP_pipeline_h5s"
h5_list = [os.path.join(h5_dir, h) for h in os.listdir(h5_dir)]

overwrite = False
failed = []
for ir, dfrow in mdf.iterrows():#df.iloc[:100].iterrows():
    try:
        #check if we've already run this session
        mouseID = str(dfrow['mouse_id'])
        opt_dir = glob.glob(os.path.join(opt_data_dir, mouseID))
        if len(opt_dir)==0:
            raise ValueError('ERROR: Could not find opt dir for {}'.format(mouseID))
        
        opt_dir = opt_dir[0]
        vis_mod_save_name = os.path.join(opt_dir, 'channel_visual_modulation_' + str(dfrow['full_id']) + '.npy')
        vis_mod_file = glob.glob(vis_mod_save_name)
        if len(vis_mod_file)>0 and not overwrite:
            print('skipping {}, vis mod data already exists'.format(dfrow['full_id']))
            continue
        
        #check if h5 file already exists for this session
        h5 = [h for h in h5_list if dfrow['full_id'] in h]
        if len(h5)>0 and False:
            h5 = h5[0]
            print('loading: {}  number {} from h5'.format(h5, ir))
            ee2 = ebs.EcephysBehaviorSession.from_h5(h5)
        
        #if not, make session from scratch
        else:
            local_dir = dfrow['path']
            print('loading: number {} from {}'.format(ir, local_dir))
            ee2 = ebs.EcephysBehaviorSession.from_local(local_dir)
        
        
        
        stim_table = ee2.stim_table
        unit_table = ee2.unit_table
    
        stim_times = stim_table.loc[(stim_table['stimulus_name'].str.contains('Natural')) &
                                    (stim_table['omitted']==False)]['Start'].values
        
        
        vm = unit_table.apply(lambda row: vis_mod(row['times'], stim_times[::10]), axis=1)                           
        unit_table['vis_mod'] = vm
        probe_vis_mod = {}
        for probe in unit_table['probe'].unique():
            channel_vis_mod = np.zeros(384)
            unit_counts = np.ones(384)
            for uir, urow in unit_table.iterrows():
                if (urow['quality']=='good') and (urow['probe']==probe) and (urow['firing_rate']>0.1):
                    peak_channel = int(urow['peak_channel'])
                    channel_vis_mod[peak_channel] += urow['vis_mod']
                    unit_counts[peak_channel] += 1
                
            channel_vis_mod = channel_vis_mod/unit_counts
            channel_vis_mod = np.convolve(channel_vis_mod, np.ones(5), 'same')/5
            probe_vis_mod[probe] = channel_vis_mod
        
        #mouseID = dfrow['mouse_id']
        #opt_dir = glob.glob(os.path.join(opt_data_dir, str(mouseID)))[0]
        np.save(vis_mod_save_name, probe_vis_mod, allow_pickle=True)
        
    except Exception as e:
        print('failed to add {} due to {}'.format(dfrow['full_id'], e))
        failed.append((dfrow['full_id'], e))
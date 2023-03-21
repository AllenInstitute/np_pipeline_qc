# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 10:43:24 2022

@author: svc_ccg
"""

import glob
import json
import os
import shutil

import numpy as np
import pandas as pd

from np_pipeline_qc.legacy.analysis import glob_file, read_json
from np_pipeline_qc.legacy.validate_local_d2_files import \
    get_inserted_probes_from_platformD1json

df = pd.read_excel(r"C:\Users\svc_ccg\ccb_onedrive\OneDrive - Allen Institute\all_np_behavior_mice.xlsx")
depth_adjustments = pd.read_excel(r"C:\Users\svc_ccg\ccb_onedrive\OneDrive - Allen Institute\VBN_probe_depth_adjustments.xlsx")

dd = df.set_index('full_id').join(depth_adjustments.set_index('full_id'), rsuffix='_depth_adjustment')

save_dir = r'\\allen\programs\braintv\workgroups\neuralcoding\corbettb\VBN_production'

failed = []
for ir, row in dd.iterrows():
    try:
        if isinstance(row['path'], str) and row['On Wayne Sheet']:
            data_dir = row['path']
            probe_dirs = glob.glob(os.path.join(data_dir, ir + '_probe*_sorted'))
            
            for probe in probe_dirs:
                probe_name = probe[-8]
                surface_channel = row[probe_name + '_surface_channel']
                if np.isnan(surface_channel):
                    raise ValueError('No surface channel for probe {} for session {}'.format(probe_name, ir))
                
                filename = glob_file(probe, 'probe_info.json')
                #filename = r"\\10.128.54.20\sd8\1093642839_553253_20210331\1093642839_553253_20210331_probeA_sorted\probe_info.json"
                if not filename:
                    raise ValueError('No probe info json found for probe {} for session {}'.format(probe_name, ir))
                
                with open(filename, 'r') as pj:
                    probe_json = json.load(pj)
                
                probe_json['mask'][191] = False
                probe_json['probe']['subprocessors'][0]['name'] = 'Neuropix-PXI-100.0'
                probe_json['probe']['subprocessors'][1]['name'] = 'Neuropix-PXI-100.1'
                probe_json['surface_channel'] = surface_channel
                
                if isinstance(row['Notes'], str) and 'Ultra' in row['Notes']:
                    raise ValueError('Ultra experiment {}, skipping'.format(ir))
                    #probe_json['probe']['phase'] = 'Ultra'
                else:
                    probe_json['probe']['phase'] = '1.0'
                
                newfilename = os.path.join(os.path.dirname(filename), 'probe_info_corrected.json')
                with open(newfilename, 'w') as pj:
                    json.dump(probe_json, pj)
                    
                
                #save to folder for Wayne
                source_dir = os.path.dirname(newfilename)
                dest_dir_name = os.path.basename(os.path.dirname(newfilename))
                dest_path = os.path.join(save_dir, dest_dir_name)
                
                if not os.path.exists(dest_path):
                    os.mkdir(dest_path)
                    
                shutil.copy(newfilename, os.path.join(dest_path, 'probe_info.json'))
    except Exception as e:
        failed.append((ir, e))
        
        
        
        
        
#VALIDATION
probe_slot_dict = {'A': '2', 'B': '2', 'C': '2', 'D': '3', 'E': '3', 'F': '3'}
probe_port_dict = {'A': '1', 'B': '2', 'C': '3', 'D': '1', 'E': '2', 'F': '3'}
pjson_dirs = os.listdir(r"\\allen\programs\braintv\workgroups\neuralcoding\corbettb\VBN production")
pjson_dirs = [os.path.join(save_dir, p) for p in pjson_dirs]
validation_failed = []
for pjdir in pjson_dirs:
    probe_name = pjdir[-8]
    try:
        pj = read_json(os.path.join(pjdir, 'probe_info.json'))
        assert pj['probe']['slot'] == probe_slot_dict[probe_name], 'slot'
        assert pj['probe']['port'] == probe_port_dict[probe_name], 'port'
        assert pj['probe']['lfp gain'] == '250x', 'lfp gain'
        assert pj['probe']['ap gain'] == '500x', 'ap gain'
        assert pj['probe']['reference channel'] == 'Tip', 'reference'
        assert pj['mask'][191] == False, 'mask'
        
        
    
    except Exception as e:
        if str(e)=='slot':
            print(pjdir)
            print(pj['probe']['slot'])
            pj['probe']['slot'] = probe_slot_dict[probe_name]
            print('correcting slot info for {}'.format(pjdir))
            analysis.save_json(pj, os.path.join(pjdir, 'probe_info.json'))
            
        validation_failed.append((pjdir, e))




for v in validation_failed:
    if str(v[1])=='reference':
        session = v[0]
        full_id = os.path.basename(session)[:26]
        row = dd.loc[full_id]
        data_dir = row['path']
        probe_name = session[-8]
        probe_dir  = glob.glob(os.path.join(data_dir, full_id + '_probe' + probe_name + '_sorted'))[0]
        times_file = os.path.join(probe_dir, 'continuous', 'Neuropix-PXI-100.0', 'spike_times.npy')

        times = np.load(times_file)
        h,b = np.histogram(times, bins=range(0, times.max(), 10000))
        fig, ax = plt.subplots()
        ax.plot(b[:-1], h)
        fig.suptitle(full_id + ' ' + probe_name)
    if str(v[1])=='slot':
        session = v[0]
        full_id = os.path.basename(session)[:26]
        row = dd.loc[full_id]
        data_dir = row['path']
        probe_name = session[-8]
        probe_dir  = glob.glob(os.path.join(data_dir, full_id + '_probe' + probe_name + '_sorted'))[0]
        times_file = os.path.join(probe_dir, 'continuous', 'Neuropix-PXI-100.0', 'spike_times.npy')

        times = np.load(times_file)
        h,b = np.histogram(times, bins=range(0, times.max(), 10000))
        fig, ax = plt.subplots()
        ax.plot(b[:-1], h)
        fig.suptitle(full_id + ' ' + probe_name)





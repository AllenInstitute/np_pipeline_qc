# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:08:13 2020

@author: svc_ccg
"""
import os, glob, re, sys
from collections import namedtuple, OrderedDict
import subprocess
import logging
import json

#source_dir = r"\\10.128.50.43\sd6.3\1047969464_509808_20200902"

rig_limsdirectory_dict = {'NP.1': r'\\W10dt05501\e', 'NP.0': r'\\W10dt05515\e'} # for now just use the NP1 directory'NP.0': r'\\W10DT05515\e'}
data_file_params = namedtuple('data_file_params',['relpath','upload','sorting_step'])

relpaths = {
        'lfp':r"continuous\Neuropix-PXI-100.1",
        'spikes':r"continuous\Neuropix-PXI-100.0",
        'events':r"events\Neuropix-PXI-100.0\TTL_1",
        'empty':""
        }
       
data_files = { 
        'probe_info.json': data_file_params(relpath='empty', upload=True, sorting_step='depth_estimation'),
        'channel_states.npy': data_file_params(relpath='events', upload=True, sorting_step='extraction'),
        'event_timestamps.npy': data_file_params(relpath='events', upload=True, sorting_step='extraction'),
        'continuous.dat': data_file_params(relpath='lfp', upload=True, sorting_step='extraction'),
        'lfp_timestamps.npy': data_file_params(relpath='lfp', upload=True, sorting_step='sorting'),
        'amplitudes.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'spike_times.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'mean_waveforms.npy': data_file_params(relpath='spikes', upload=True, sorting_step='mean waveforms'),
        'spike_clusters.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'spike_templates.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'templates.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'whitening_mat.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'whitening_mat_inv.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'templates_ind.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'similar_templates.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'metrics.csv': data_file_params(relpath='spikes', upload=True, sorting_step='metrics'),
        'channel_positions.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
#        'cluster_group.tsv': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'channel_map.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'params.py': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'probe_depth.png': data_file_params(relpath='empty', upload=True, sorting_step='depth estimation'),
        }
#        'continuous\\Neuropix-3a-100.0\\continuous.dat': data_file_params(relpath='empty', upload=False, sorting_step='extraction'),
#        'residuals.dat': data_file_params(relpath='spikes', upload=False, sorting_step='median subtraction'),
#        'pc_features.npy': data_file_params(relpath='spikes', upload=False, sorting_step='sorting'),
#        'template_features.npy': data_file_params(relpath='spikes', upload=False, sorting_step='sorting'),
#        'rez2.mat': data_file_params(relpath='spikes', upload=False, sorting_step='sorting'),
#        'rez.mat': data_file_params(relpath='spikes', upload=False, sorting_step='sorting'),
#        'pc_feature_ind.npy': data_file_params(relpath='spikes', upload=False, sorting_step='sorting'),
#        'template_feature_ind.npy': data_file_params(relpath='spikes', upload=False, sorting_step='sorting')
#        }


def transfer_session(session_base_dir, probes_to_run = 'ABCDEF', rig_dir_dict=None):
    return_string = []
    try: 
        rig = get_rig(session_base_dir)
        if rig_dir_dict:
            lims_dir = rig_dir_dict[rig]
        elif all(folder in session_base_dir for folder in ['workgroups', 'mindscope', 'np-exp']):
            lims_dir = r"\\allen\programs\mindscope\workgroups\np-exp\outbox"
        else:
            lims_dir = rig_limsdirectory_dict[rig]
        
        probe_dirs = get_probe_directories(session_base_dir)
        
        #FIRST VALIDATE THAT ALL FILES ARE PRESENT FOR ALL PROBES; DONT TRANSFER PARTIAL SESSIONS
        for pd in probe_dirs:   
            pid = get_probe_id_from_dir(pd)
            if pid in probes_to_run:
                file_dict = validate_d2_files(pd)
                missing = [not(f['exists']) for _,f in file_dict.items()]
                if any(missing):
                    out = 'Must have all D2 files before transfer can take place'
                    print(out)
                    return_string.append(out)
                    return return_string

        #IF ALL SESSIONS LOOK GOOD, START TRANSFER            
        for pd in probe_dirs:   
            pid = get_probe_id_from_dir(pd)
            if pid in probes_to_run:
                file_dict = validate_d2_files(pd)
                print(pd)
                return_string.append('Probe Directory {}'.format(pd))
                p_dest_dir = os.path.join(lims_dir, os.path.basename(pd))
                print(p_dest_dir)
                return_string.append('Destination Directory {}\n'.format(p_dest_dir))
                transfer_d2_files(pd, p_dest_dir, file_dict)
    except:
        for ind in [0,1,2]:
            return_string.append(str(sys.exc_info()[ind]))
    
    return return_string    

def get_probe_id_from_dir(dirname):
    
    base_str = os.path.basename(dirname)
    base_str_parts = base_str.split('_')
    probe_part = [part for part in base_str_parts if 'probe' in part][0]
    pid = re.match('probe[A-F]', probe_part).group(0)[-1]
    
    return pid


def transfer_d2_files(session_base_dir, dest_dir, file_dict):
    
    transfer_dict = get_file_transfer_dict(session_base_dir, dest_dir, file_dict)
    for filename, params in data_files.items():
        info = transfer_dict[filename]
        source_dir = os.path.dirname(info['source'])
        dest_dir = os.path.dirname(info['dest'])
        
        real_filename = os.path.basename(info['source'])
        
        command_string = "robocopy "+ source_dir +" " + dest_dir + " " + real_filename + r" /xc /xn /xo"
        print(command_string)
        P = subprocess.call(command_string)
        print('Copied {} with return code {}'.format(real_filename, P))


def validate_d2_files(session_base_dir):
    
    file_dict = {f:{'path':None, 'exists':False} for f in data_files}
    base = session_base_dir
    
    for filename, params in data_files.items():
        
        rel_path_indexer = params.relpath
        file_rel_path = relpaths[rel_path_indexer]
        file_dir_abs = os.path.join(base, file_rel_path)
        
        if filename == 'probe_depth.png':
            glob_filename = 'probe_depth*.png'
        else:
            glob_filename = filename
        
        real_path = glob_file(file_dir_abs, glob_filename)
        if real_path is not None:
            file_dict[filename]['exists'] = True
            file_dict[filename]['path'] = real_path

    return file_dict


def get_file_transfer_dict(session_base_dir, dest_dir, file_dict):
    
    transfer_dict = {f:{'source': None, 'dest': None} for f in data_files}
    
    for filename, params in data_files.items():
        
        source_path = file_dict[filename]['path']
        transfer_dict[filename]['source'] = source_path
        
        real_filename = os.path.basename(source_path)        
        transfer_dict[filename]['dest'] = make_dest_path(dest_dir, relpaths[data_files[filename].relpath], real_filename)
    
    return transfer_dict
        

def make_dest_path(dest_dir, rel_path, filename):
    
    dest_base = os.path.join(dest_dir, rel_path)
    dest_path = os.path.join(dest_base, filename)
    
    return dest_path


def get_probe_directories(base):
    
    pattern = re.compile('1[0-9]{9}_[0-9]{6}_[0-9]{8}_probe[A-F]_sorted')
    
    probe_dirs = glob.glob(os.path.join(base, '[0-9]*probe*_sorted'))
    good_dirs = []
    for p in probe_dirs:
        pbase = os.path.basename(p)
        match = pattern.match(pbase)
        
        if match.group(0) is not None:
            good_dirs.append(p)
        
        
    good_dirs = [p for p in good_dirs if os.path.isdir(p)]
    
    return probe_dirs

def get_rig(session_base_dir):
    
    platform_json = glob_file(session_base_dir, '*platformD1.json')
    
    with open(platform_json, 'r') as file:
        platform_data = json.load(file)
        
    rig = platform_data['rig_id']
    
    return rig        

    
def glob_file(root, format_str):
    
    f = glob.glob(os.path.join(root, format_str))
    if len(f)>0:
        return f[0]
    else:
        print('Could not find file of format '\
              '{} in {}'.format(format_str, root))
        return None








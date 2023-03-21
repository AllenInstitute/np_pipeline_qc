# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 17:21:26 2020

@author: svc_ccg
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 19:09:46 2020

@author: svc_ccg
"""

import os

import pandas as pd
from matplotlib import pyplot as plt

import np_pipeline_qc.legacy.get_sessions as gs
from np_pipeline_qc.legacy.run_qc_class import run_qc_hab

#TODO: LOGGING!!! 

sources = [r"\\10.128.50.43\sd6.3\habituation", r"\\10.128.50.20\sd7\habituation", 
           r"\\10.128.50.20\sd7.2\habituation", r"\\10.128.54.20\sd8\habituation",
           r"\\10.128.54.20\sd8.2\habituation", r"\\10.128.50.43\sd6.2\habituation"]
sessions_to_run = gs.get_sessions(sources, mouseID='!366122', start_date='20210510')#, end_date='20201101')
destination = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\QC\habituation"
modules_to_run = ['vsyncs', 'behavior', 'probeSyncAlignment']
cortical_sort = False

just_run_new_sessions = True
run_only_missing_modules = False

def find_new_sessions_to_run(sessions_to_run, destination):
    all_session_ids = [os.path.split(s)[-1] for s in sessions_to_run]
    
    dest_sessions = gs.get_sessions(destination)
    dest_session_ids = [os.path.split(s)[-1] for s in dest_sessions]
    
    return [sessions_to_run[i] for i, d in enumerate(all_session_ids) if d not in dest_session_ids]


def get_missing_modules(sessions_to_run, module_list):
    
    #if all modules are selected, populate list
    if module_list == 'all':
        module_list = [p for p in dir(run_qc_hab) if not p[0]=='_']
    
    ignore_list = ['data_loss'] #hack since data_loss if part of probe_yield
    session_missing_modules = {}
    for s in sessions_to_run:
        base = os.path.basename(s)
        qc_dirname = os.path.join(destination, base)
        qc_dirs = os.listdir(qc_dirname)
        missing_modules = []
        for m in module_list:
            if m not in qc_dirs and m not in ignore_list:
                missing_modules.append(m)
        
        session_missing_modules[s] = missing_modules
    
    return session_missing_modules
    

if just_run_new_sessions:
    sessions_to_run = find_new_sessions_to_run(sessions_to_run, destination)

if run_only_missing_modules:
    session_missing_modules = get_missing_modules(sessions_to_run, modules_to_run)


failed = []
sync_problem_sessions = []
session_errors = {}
for ind, s in enumerate(sessions_to_run):
    
    session_name = os.path.basename(s)
    session_modules_to_run = session_missing_modules[s] \
        if run_only_missing_modules else modules_to_run

    print('\nRunning modules {} for session {}, {} in {} \n'
          .format(session_modules_to_run,session_name, 
                  ind+1, len(sessions_to_run)))
    
    try:
        
        r=run_qc_hab(s, destination, modules_to_run=session_modules_to_run, cortical_sort=cortical_sort)
        
        if len(r.vf)<r.total_pkl_frames:
            print('found bad sync data')
            sync_problem_sessions.append([s, r.platform_info['rig_id'], r.platform_info['ExperimentStartTime']])
        
        session_errors[s] = r.errors
        #pd.to_pickle(r.probe_dict, os.path.join(local_probe_dict_save_dir, session_name+'_unit_table.pkl'))
    
    except Exception as e:
        failed.append((s, e))
        print('Failed to run session {}, due to error {} \n'
              .format(session_name, e))
    plt.close('all')
        

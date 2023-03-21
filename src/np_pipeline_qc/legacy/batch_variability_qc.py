# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 17:39:33 2022

@author: svc_ccg
"""
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import np_pipeline_qc.legacy.get_sessions as gs
from np_pipeline_qc.legacy.run_qc_class import run_qc_passive

#TODO: LOGGING!!! 

sources = [r"\\10.128.50.43\sd6.3", r"\\10.128.50.20\sd7", r"\\10.128.50.20\sd7.2", 
           r"\\10.128.54.20\sd8", r"\\10.128.54.20\sd8.2", r"\\10.128.54.20\sd8.3",
           r"\\10.128.54.19\sd9"]

variability_mice = [
        '576325',
        '576321',
        '578002',
        '578004',
        '594585',
        '594584',
        '594534',
        '593788',
        '597503',
        '597504',
        '597507',
        '597505',
        '598431',
        '597506',
        '599894',
        '602518']


mouse_string = '$' + '$'.join(variability_mice)
sessions_to_run = gs.get_sessions(sources, mouseID=mouse_string, start_date='20200601')#, end_date='20200930')
#filter out duplicate sessions
session_ids = np.array([os.path.basename(s)[:10] for s in sessions_to_run])
sess_to_keep = []
for spath, sid in zip(sessions_to_run, session_ids):
    sessions = np.where(session_ids==sid)[0]
    session_paths = [sessions_to_run[sind] for sind in sessions]
    if len(sessions)>1:
        #take the session folder with more stuff in it
        sizes = []
        for sess in session_paths:
            size = len([files for root, dirs, files in os.walk(sess)])
            sizes.append(size)
        
        sess_to_keep.append(session_paths[np.argmax(sizes)])
    else:
        sess_to_keep.append(spath)

sessions_to_run = [s for s in sessions_to_run if s in sess_to_keep]


destination = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\variability"
modules_to_run = 'all' #['probe_targeting', 'behavior']
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
        module_list = [p for p in dir(run_qc_passive) if not p[0]=='_']
    
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
session_errors = {}
for ind, s in enumerate(sessions_to_run):
    
    session_name = os.path.basename(s)
    session_modules_to_run = session_missing_modules[s] \
        if run_only_missing_modules else modules_to_run

    print('\nRunning modules {} for session {}, {} in {} \n'
          .format(session_modules_to_run,session_name, 
                  ind+1, len(sessions_to_run)))
    
    try:
        
        r=run_qc_passive(s, destination, modules_to_run=session_modules_to_run, cortical_sort=cortical_sort)
        session_errors[s] = r.errors
        #pd.to_pickle(r.probe_dict, os.path.join(local_probe_dict_save_dir, session_name+'_unit_table.pkl'))
    
    except Exception as e:
        failed.append((s, e))
        print('Failed to run session {}, due to error {} \n'
              .format(session_name, e))
    plt.close('all')




# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 19:09:46 2020

@author: svc_ccg
"""

import get_sessions as gs
import os
from run_qc_class import run_qc
from matplotlib import pyplot as plt
import pandas as pd

#TODO: LOGGING!!! 

sources = [r"\\10.128.50.43\sd6.3", r"\\10.128.50.20\sd7"]
sessions_to_run = gs.get_sessions(sources, mouseID='!366122', start_date='20200601')#, end_date='20200922')
destination = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\mochi"

local_probe_dict_save_dir = r"C:\Data\NP_behavior_unit_tables"
just_run_new_sessions = True

def find_new_sessions_to_run(sessions_to_run, destination):
    all_session_ids = [os.path.split(s)[-1] for s in sessions_to_run]
    
    dest_sessions = gs.get_sessions(destination)
    dest_session_ids = [os.path.split(s)[-1] for s in dest_sessions]
    
    return [sessions_to_run[i] for i, d in enumerate(all_session_ids) if d not in dest_session_ids]

if just_run_new_sessions:
    sessions_to_run = find_new_sessions_to_run(sessions_to_run, destination)

modules_to_run = 'all' #['probe_targeting', 'behavior']
cortical_sort = False
failed = []
session_errors = {}
for ind, s in enumerate(sessions_to_run):
    
    session_name = os.path.basename(s)
    print('\nRunning QC for session {}, {} in {} \n'
          .format(session_name, ind+1, len(sessions_to_run)))
    
    try:
        r=run_qc(s, destination, modules_to_run=modules_to_run, cortical_sort=cortical_sort)
        session_errors[s] = r.errors
        pd.to_pickle(r.probe_dict, os.path.join(local_probe_dict_save_dir, session_name+'_unit_table.pkl'))
    
    except Exception as e:
        failed.append((s, e))
        print('Failed to run session {}, due to error {} \n'
              .format(session_name, e))
    plt.close('all')
        








failed_sessions = [   
    '\\\\10.128.50.43\\sd6.3\\1028043324_498757_20200604',
     '\\\\10.128.50.43\\sd6.3\\1028225380_498757_20200605',
     '\\\\10.128.50.43\\sd6.3\\1029247206_498803_20200610',
     '\\\\10.128.50.43\\sd6.3\\1030489628_498756_20200617',
     '\\\\10.128.50.43\\sd6.3\\1030680600_498756_20200618',
     '\\\\10.128.50.43\\sd6.3\\1031938107_485124_20200624',
     '\\\\10.128.50.43\\sd6.3\\1032143170_485124_20200625',
     '\\\\10.128.50.43\\sd6.3\\1033387557_509940_20200630',
     '\\\\10.128.50.43\\sd6.3\\1033388795_509652_20200630',
     '\\\\10.128.50.43\\sd6.3\\1033611657_509652_20200701',
     '\\\\10.128.50.43\\sd6.3\\1034912109_512913_20200708',
     '\\\\10.128.50.43\\sd6.3\\1036476611_506798_20200715',
     '\\\\10.128.50.43\\sd6.3\\1036675699_506798_20200716',
     '\\\\10.128.50.43\\sd6.3\\1037747248_505167_20200721',
     '\\\\10.128.50.43\\sd6.3\\1037927382_513573_20200722',
     '\\\\10.128.50.43\\sd6.3\\1038127711_513573_20200723',
     '\\\\10.128.50.43\\sd6.3\\1039557143_524921_20200730',
     '\\\\10.128.50.43\\sd6.3\\1043752325_506940_20200817',
     '\\\\10.128.50.43\\sd6.3\\1044016459_506940_20200818',
     '\\\\10.128.50.43\\sd6.3\\1044026583_509811_20200818',
     '\\\\10.128.50.43\\sd6.3\\1044385384_524761_20200819',
     '\\\\10.128.50.43\\sd6.3\\1046651551_527294_20200827',
     '\\\\10.128.50.43\\sd6.3\\2033616558_509940_20200701',
     '\\\\10.128.50.43\\sd6.3\\2041083421_522944_20200805']
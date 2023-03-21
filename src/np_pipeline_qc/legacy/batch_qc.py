# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 19:09:46 2020

@author: svc_ccg
"""

import json
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import np_pipeline_qc.legacy.get_sessions as gs
from np_pipeline_qc.legacy.run_qc_class import run_qc

# TODO: LOGGING!!!

# sources = [r"\\10.128.50.43\sd6.3", r"\\10.128.50.20\sd7", r"\\10.128.50.20\sd7.2",
#           r"\\10.128.54.20\sd8", r"\\10.128.54.20\sd8.2", r"\\10.128.54.20\sd8.3",
#           r"\\10.128.54.19\sd9"]
source_volume_config = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\source_list.json'
with open(source_volume_config, 'r') as f:
    sources = json.load(f)

mice_to_skip = '!366122!544480!576325!576321!578002!578004!594585!594584!594534!593788!597503!597504!597507!597505!598431!597506'
sessions_to_run = gs.get_sessions(
    sources, mouseID=mice_to_skip, start_date='20200601'
)  # , end_date='20200930')
# filter out duplicate sessions
session_ids = np.array([os.path.basename(s)[:10] for s in sessions_to_run])
sess_to_keep = []
for spath, sid in zip(sessions_to_run, session_ids):
    sessions = np.where(session_ids == sid)[0]
    session_paths = [sessions_to_run[sind] for sind in sessions]
    if len(sessions) > 1:
        # take the session folder with more stuff in it
        sizes = []
        for sess in session_paths:
            size = len([files for root, dirs, files in os.walk(sess)])
            sizes.append(size)

        sess_to_keep.append(session_paths[np.argmax(sizes)])
    else:
        sess_to_keep.append(spath)

sessions_to_run = [s for s in sessions_to_run if s in sess_to_keep]


destination = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\mochi'
modules_to_run = 'all'   # ['probe_targeting', 'behavior']
cortical_sort = False

local_probe_dict_save_dir = r'C:\Data\NP_behavior_unit_tables'
just_run_new_sessions = True
run_only_missing_modules = False


def find_new_sessions_to_run(sessions_to_run, destination):
    all_session_ids = [os.path.split(s)[-1] for s in sessions_to_run]

    dest_sessions = gs.get_sessions(destination)
    dest_session_ids = [os.path.split(s)[-1] for s in dest_sessions]

    return [
        sessions_to_run[i]
        for i, d in enumerate(all_session_ids)
        if d not in dest_session_ids
    ]


def get_missing_modules(sessions_to_run, module_list):

    # if all modules are selected, populate list
    if module_list == 'all':
        module_list = [p for p in dir(run_qc) if not p[0] == '_']

    ignore_list = ['data_loss']   # hack since data_loss if part of probe_yield
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
    session_missing_modules = get_missing_modules(
        sessions_to_run, modules_to_run
    )

# sessions_to_run = [r'\\10.128.54.20\sd8.3\1130113579_579993_20210922',
#                   r'\\10.128.54.20\sd8.2\1101263832_563326_20210505',
#                   r'\\10.128.54.20\sd8.2\1104058216_563497_20210519',
#                   r'\\10.128.54.20\sd8.2\1102790314_567284_20210513',
#                   r'\\10.128.50.20\sd7\1053709239_532246_20200930',
#                   r'\\10.128.54.20\sd8.2\1101467873_563326_20210506']


failed = []
session_errors = {}
for ind, s in enumerate(sessions_to_run):

    session_name = os.path.basename(s)
    session_modules_to_run = (
        session_missing_modules[s]
        if run_only_missing_modules
        else modules_to_run
    )

    print(
        '\nRunning modules {} for session {}, {} in {} \n'.format(
            session_modules_to_run, session_name, ind + 1, len(sessions_to_run)
        )
    )

    try:

        r = run_qc(
            s,
            destination,
            modules_to_run=session_modules_to_run,
            cortical_sort=cortical_sort,
        )
        session_errors[s] = r.errors
        # pd.to_pickle(r.probe_dict, os.path.join(local_probe_dict_save_dir, session_name+'_unit_table.pkl'))

    except Exception as e:
        failed.append((s, e))
        print(
            'Failed to run session {}, due to error {} \n'.format(
                session_name, e
            )
        )
    plt.close('all')


# failed = []
# for s in sessions_to_run:
#
#    try:
#        r=run_qc(s, destination, modules_to_run='none', cortical_sort=cortical_sort)
#        r._make_session_meta_json()
#    except:
#        failed.append(s)


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
    '\\\\10.128.50.43\\sd6.3\\2041083421_522944_20200805',
]

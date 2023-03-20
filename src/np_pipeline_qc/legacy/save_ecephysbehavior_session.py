# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 15:48:00 2021

@author: svc_ccg
"""

import get_sessions as gs
import os
import EcephysBehaviorSession as ebs
import pandas as pd
import shutil

#TODO: LOGGING!!! 

sources = [r"\\10.128.50.43\sd6.3", r"\\10.128.50.20\sd7", r"\\10.128.50.20\sd7.2", 
           r"\\10.128.54.20\sd8", r"\\10.128.54.20\sd8.2", r"\\10.128.54.20\sd8.3",
           r"\\10.128.54.19\sd9"]
#sessions_to_run = gs.get_sessions(sources, mouseID='!366122', start_date='20200930')#, end_date='20200922')

#sessions_to_run = gs.get_sessions(sources, mouseID='530862')
sessions_to_run = gs.get_sessions(sources, limsID='1090803859')
destination = r"C:\Data\NP_pipeline_h5s"

no_space = []
failed = []
for s in sessions_to_run:
    
    free_space = shutil.disk_usage(destination)[2]/1e9
    if free_space<10:
        print('not enough space for {}'.format(s))
        no_space.append(s)
        continue
        
    try:
        e = ebs.EcephysBehaviorSession.from_local(s)
        save_base = os.path.basename(s)
        save_path = os.path.join(destination, save_base + '.h5')
        ebs.save_to_h5(e, save_path)
        del(e)
    
    except:
        print('failed to load {}'.format(s))
        failed.append(s)
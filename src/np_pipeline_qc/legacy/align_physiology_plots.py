# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 17:27:45 2022

@author: svc_ccg
"""

import sys
sys.path.append(r"C:\Users\svc_ccg\Documents\GitHub\AIBSOPT_VBN\Software\Analysis")
from align_to_physiology import align_to_physiology


import EcephysBehaviorSession as ebs
import os, glob
import pandas as pd
import analysis
import numpy as np

df = pd.read_excel(r"C:\Users\svc_ccg\ccb_onedrive\OneDrive - Allen Institute\all_np_behavior_mice.xlsx")
opt_data_dir = r'\\allen\programs\mindscope\workgroups\np-behavior\processed_ALL'

failed = []
count = 0
for ir, row in mdf.iterrows():
    try:
        print('running {}, {} of {}'.format(row['mouse_id'], count, len(df)))
        mouseID = str(int(row['mouse_id']))
        local_dir = row['path']
        opt_dir = glob.glob(os.path.join(opt_data_dir, mouseID))
        if len(opt_dir)==0:
            raise ValueError('ERROR: Could not find opt dir for {}'.format(mouseID))
        
        opt_dir = opt_dir[0]
        align_to_physiology(local_dir, opt_dir)
    
    except Exception as e:
        print('failed on session {} with error {}'.format(local_dir, e))
        failed.append((local_dir, e))
        

    finally:
        count+=1











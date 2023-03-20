# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:57:05 2021

@author: svc_ccg
"""

from run_qc_class import run_qc
from matplotlib import pyplot as plt
import pandas as pd
import get_sessions as gs
import json, os, glob
import probeSync_qc as probeSync

#TODO: LOGGING!!! 

sources = [r"\\10.128.50.43\sd6.3", 
           r"\\10.128.50.20\sd7", r"\\10.128.50.20\sd7.2", 
           r"\\10.128.54.20\sd8", r"\\10.128.54.20\sd8.2"]
#qc_dirs = [os.path.join(destination, os.path.basename(s)) for s in sessions_to_run]

probe_info = {}

probe_changes = []
ext_ref = []
for rig in ['NP0', 'NP1']:
    sessions = gs.get_sessions(sources, mouseID='!366122', start_date='20210101', rig=rig)#, end_date='20200922')
    last_info = None
    
    for sess in sessions:
        try:
            xmlfiledir = glob.glob(os.path.join(sess, '*probeABC'))[0]
            xmlfilepath = glob.glob(os.path.join(xmlfiledir, 'settings.xml'))[0]
            
            pinfo = probeSync.get_probe_settings_from_xml(xmlfilepath)
            if last_info is not None:
                for probe in 'ABCDEF':
                    if not pinfo[probe]['probe_serial_number']==last_info[probe]['probe_serial_number']:
                        probe_changes.append([sess, rig, probe, pinfo[probe]['probe_serial_number']])
                    
                    if not pinfo[probe]['referenceChannel'] == 'Tip':
                        ext_ref.append([sess, rig, probe])
                        
            last_info = pinfo
        except:
            print('failed to get settings file for {}'.format(sess))
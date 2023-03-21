# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:38:00 2020

@author: svc_ccg
"""

import datetime
import os
import re

import np_pipeline_qc.legacy.get_sessions as gs
import np_pipeline_qc.legacy.lims_validation as lv
from np_pipeline_qc.legacy.lims_validation import lims_validation

#TODO: LOGGING!!! 

rigs_to_check = ['NP1', 'NP0']
#source = r"\\10.128.50.43\sd6.3"
#dest = r"\\10.128.50.43\sd6.3\lims validation"
sources = [r"\\10.128.54.20\sd8.3", r"\\10.128.54.20\sd8.2", r"\\10.128.54.20\sd8", 
            r"\\10.128.50.20\sd7.2", r"\\10.128.50.20\sd7", 
            r"\\10.128.50.43\sd6.3"]
dest = os.path.join(r"\\10.128.50.20\sd7", 'lims_validation')

if not os.path.exists(dest):
    os.mkdir(dest)
    
    
def get_lims_id_from_session_dir(sdir):
    
    base = os.path.basename(sdir)
    lims_id = re.search('[0-9]{10}', base).group(0)
    
    return lims_id


for rig in rigs_to_check:
    
    sessions_to_run = []
    for source in sources:
        sessions = gs.get_sessions(source, mouseID='!366122', rig=rig, start_date='20200101')
        if len(sessions)==1:
            sessions_to_run.append(sessions)
        elif len(sessions)>1:
            sessions_to_run.extend(sessions
                                   )
    #destination = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\mochi"
     
    status = {}
    D1_to_run = {}
    D2_to_run = {}
    D1_to_check = {}
    D2_to_check = {}
    complete = {}
    for session in sessions_to_run:
        
        lims_id = get_lims_id_from_session_dir(session)  
        save_path = os.path.join(session, 'lims_upload_report_'+str(lims_id)+'.json')  
        report = lv.run_validation(lims_id, save_path)  
        session_name = os.path.basename(session)
    
        if not report['D1_upload_summary']['upload_exists']:
            D1_to_run[session] = report['file_validation']['Error']
        elif not report['D2_upload_summary']['upload_exists']:
            D2_to_run[session] = report['file_validation']['Error']
    
        if report['D1_upload_summary']['upload_exists'] and not report['D1_upload_summary']['pass']:
            D1_to_check[session] = report['D1_upload_summary']['errors']
        
        if report['D2_upload_summary']['upload_exists'] and not report['D2_upload_summary']['pass']:
            D2_to_check[session] = report['D2_upload_summary']['errors']
           
        status[session_name] = report
        
        if (report['D1_upload_summary']['upload_exists'] and report['D1_upload_summary']['pass']
            and report['D2_upload_summary']['upload_exists'] and report['D2_upload_summary']['pass']):
            complete[session] = 'upload complete!'
    
    overall_summary = {
                'D1_to_run':D1_to_run,
                'D2_to_run':D2_to_run,
                'D1_to_check': D1_to_check,
                'D2_to_check': D2_to_check,
                'complete': complete,
                'session_details':status}   
    now = datetime.datetime.now()
    now_string = now.strftime('%Y%m%d%H%M%S')
    lv.save_json(overall_summary, os.path.join(dest, rig+'_lims_upload_status_' + now_string+'.json'))
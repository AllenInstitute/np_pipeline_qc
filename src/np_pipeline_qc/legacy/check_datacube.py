# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 09:46:23 2021

@author: svc_ccg
"""

from run_qc_class import run_qc
from matplotlib import pyplot as plt
import pandas as pd
import get_sessions as gs
import json, os, glob
import numpy as np
from lims_validation import run_validation
from query_lims import query_lims

#TODO: LOGGING!!! 
source_volume_config = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\source_list.json"
with open(source_volume_config, 'r') as f:
    sources = json.load(f)
    
#sources = [r"\\10.128.50.43\sd6.3", r"\\10.128.50.20\sd7", r"\\10.128.50.20\sd7.2", 
#           r"\\10.128.54.20\sd8", r"\\10.128.54.20\sd8.2", r"\\10.128.54.20\sd8.3",
#           r"\\10.128.54.19\sd9"]

#omit test mice and passive mice
mice_to_skip = '!366122!544480!576325!576321!578002!578004!594585!594584!594534!593788!597503!597504!597507!597505!598431'

sessions_to_run = gs.get_sessions(sources, mouseID=mice_to_skip, start_date='20200715')#, end_date='20200930')
destination = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\mochi"

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


# make data frame to summarize current status
def get_genotype(lims_id):
    query_string = '''
        SELECT es.id as es_id, sp.name as specimen_name
        FROM ecephys_sessions es
        JOIN specimens sp ON sp.id = es.specimen_id
        WHERE es.id = {}
        ORDER BY es.id
        '''
    try:
        genotype_info = query_lims(query_string.format(lims_id))
        if len(genotype_info)>0 and 'specimen_name' in genotype_info[0]:
            genotype_string = genotype_info[0]['specimen_name']
            genotype = genotype_string[:genotype_string.rfind('-')]
        else:
            print('Could not find genotype for session {}'.format(lims_id))
            genotype = ''
    except Exception as e:
        genotype = ''
        print('Error retrieving genotype: {}'.format(e))
    return genotype


def get_session_meta(path):
    
    platform_json = glob.glob(os.path.join(path, '*platformD1.json'))
    if len(platform_json)==0:
        operator = ''
        rig = ''
        stimulus = ''
    
    else:
        with open(platform_json[0], 'r') as f:
            info = json.load(f)
            operator = info.get('operatorID', '')
            rig = info.get('rig_id', '')
            stimulus = info.get('stimulus_name', '')
   
    return operator, rig, stimulus

def sort_columns(dataframe, ordered_cols):
    '''Rearrage columns in dataframe.
        INPUT:
            dataframe: the dataframe you want to resort
            ordered_cols: order of the columns you want.
                You can specify a subset of the total columns here,
                and the function will just tack on the rest ordered
                alphanumerically
    '''
    all_cols = dataframe.columns.tolist()
    
    #if there are more columns than specified in
    #ordered_cols, just tack them on alphabetically
    if len(ordered_cols) < len(all_cols):
        sorted_cols = [c for c in np.sort(all_cols) if c not in ordered_cols]
        final_cols = ordered_cols + sorted_cols
    else:
        final_cols = ordered_cols
    
    return dataframe[final_cols]

create_new = False
if create_new:
    df = pd.DataFrame(sessions_to_run, columns=['path'])  #MAKE NEW
    df['mouse_id'] = df.apply(lambda x: x['path'].split('_')[1], axis=1)
    df['lims_id'] = df.apply(lambda x: x['path'].split('_')[0][-10:], axis=1)
    df['session_date'] = df.apply(lambda x: x['path'].split('_')[-1], axis=1)
    df['full_id'] = df.apply(lambda x: os.path.basename(x['path']), axis=1)
    df['QC pass'] = None
    df['OPT pass'] = None
    #df['QC reviewer 1'] = df['operator']
    df['QC reviewer 2'] = 'Sev'
    df['QC reviewer 3'] = 'Corbett'
    df['Notes'] = None
    df['Production'] = None
    df['Seizure'] = None


df = pd.read_excel(r"C:\Users\svc_ccg\ccb_onedrive\OneDrive - Allen Institute\all_np_behavior_mice.xlsx") #LOAD EXISTING
for index, row in df.iterrows():
    
    lims_validation = run_validation(row['lims_id'])
    D1 = lims_validation['D1_upload_summary'].get('pass') if lims_validation['D1_upload_summary'].get('pass') else False
    D2 = lims_validation['D2_upload_summary'].get('pass') if lims_validation['D2_upload_summary'].get('pass') else False
    
    if not D1:
        d1errors = lims_validation['D1_upload_summary'].get('errors', ['upload does not exist'])
        error_string = ''
        for error in d1errors:
            error_string = error_string + error + '; '
        error_string = error_string[:-2]
        df.at[index, 'LIMS Notes'] = 'D1: ' + error_string
    
    df.at[index, 'D1_final'] = D1
    df.at[index, 'D2_final'] = D2
    df.at[index, 'genotype'] = get_genotype(row['lims_id'])

    meta = get_session_meta(row['path'])
    df.at[index, 'operator'] = meta[0]
    df.at[index, 'QC reviewer 1'] = meta[0]
    df.at[index, 'rig'] = meta[1]
    df.at[index, 'stimulus'] = meta[2]
    df.at[index, 'storage_directory'] = lims_validation['file_validation'].get('storage_directory', '').replace('\\', '\\\\', 1)


df.loc[df['genotype'].str.contains('C57'), 'genotype'] = 'C57BL6J'
df_sorted = df.sort_values(['genotype', 'mouse_id', 'session_date'])
df_sorted = sort_columns(df_sorted, ['genotype', 'full_id', 'path', 'storage_directory', 'D1', 'D2', 'LIMS Notes'])


#format excel sheet
writer = pd.ExcelWriter(r"C:\Users\svc_ccg\ccb_onedrive\OneDrive - Allen Institute\all_np_behavior_mice_11302021.xlsx", engine='xlsxwriter')
df_sorted.to_excel(writer, sheet_name='Sheet1', index=False)
workbook  = writer.book
worksheet = writer.sheets['Sheet1']
red_format = workbook.add_format({'bg_color':   '#FFC7CE',
                               'font_color': '#9C0006'})
green_format = workbook.add_format({'bg_color':   '#C6EFCE',
                               'font_color': '#006100'})

c57_format = workbook.add_format({'bg_color':   '#EDEDED'})
sst_format = workbook.add_format({'bg_color':   '#F8CBAD'})
vip_format = workbook.add_format({'bg_color':   '#D9E1F2'})

#gray_format = workbook.add_format({'bg_color': 'gray'})
for col in ['D1', 'D2']:
    worksheet.conditional_format(1, df_sorted.columns.get_loc(col), len(df_sorted.index), df_sorted.columns.get_loc(col),
                                         {'type': 'cell',
                                          'criteria': '==',
                                           'value':     False,
                                           'format': red_format})
    
    worksheet.conditional_format(1, df_sorted.columns.get_loc(col), len(df_sorted.index), df_sorted.columns.get_loc(col),
                                         {'type': 'cell',
                                          'criteria': '==',
                                           'value':     True,
                                           'format': green_format})

start_row = 0
for g, f in zip(['C57', 'Sst', 'Vip'], [c57_format, sst_format, vip_format]):
    
    rows_to_change = np.arange(start_row, start_row+df_sorted['genotype'].str.contains(g).sum())+1
    for r in rows_to_change:
        worksheet.set_row(r, None, f)
    
    start_row += len(rows_to_change)
    
writer.save()

df_sorted.to_excel(r"C:\Users\svc_ccg\Desktop\test.xlsx", index=False)



qc_dirs = [os.path.join(destination, os.path.basename(s)) for s in sessions_to_run]

[os.path.exists(q) for q in qc_dirs]

mid_dict = {}
session_genotypes = {}
mid_genotype = {}
for qd in qc_dirs:
    
    if os.path.exists(qd):
        
        #get specimen meta json
        specimen_meta = glob.glob(os.path.join(qd, 'specimen_meta.json'))
        if len(specimen_meta)>0:
            
            with open(specimen_meta[0], 'r') as file:
                specimen_info = json.load(file)
                mid = specimen_info['mid']
                
                session_genotypes[os.path.basename(qd)] = {'mid': specimen_info['mid'],
                                                          'genotype': specimen_info['genotype']} 
                mid_genotype[specimen_info['mid']] = specimen_info['genotype']
                
                if mid in mid_dict:
                    mid_dict[mid]['sessions'].append(os.path.basename(qd))
                else:
                    mid_dict[mid] = {'genotype': specimen_info['genotype'], 'sessions':[os.path.basename(qd)]}



for geno in ['C57BL6J', 'Sst-IRES-Cre;Ai32', 'Vip-IRES-Cre;Ai32']:
    print(geno)
    for mid in mid_dict:
        
        if mid_dict[mid]['genotype'] == geno:
            print(mid)#, mid_dict[mid]['sessions'])


for geno in ['C57BL6J', 'Sst-IRES-Cre;Ai32', 'Vip-IRES-Cre;Ai32']:
    print(geno)
    for mid in mid_dict:
        
        if mid_dict[mid]['genotype'] == geno:
            print(mid_dict[mid]['sessions'][0])


for geno in ['C57BL6J', 'Sst-IRES-Cre;Ai32', 'Vip-IRES-Cre;Ai32']:
    print(geno)
    for mid in mid_dict:
        
        if mid_dict[mid]['genotype'] == geno:
            print(mid_dict[mid]['sessions'][-1])
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:22:24 2022

@author: svc_ccg
"""
import pandas as pd
import numpy as np
import decimal  # bleh
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import url
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session
from contextlib import contextmanager
import glob, os, json

ADAPTER = "sqlserver"
DATASERVER = "AILABTRACKS\SQLEXPRESS"
USERNAME = "limstracks"
PASSWORD = "m0usetr@ck"
DATABASE = "LabTracks_AIBS"
TABLE = "Animals"



df = pd.read_excel(r"C:\Users\svc_ccg\ccb_onedrive\OneDrive - Allen Institute\all_np_behavior_mice.xlsx")

b = pd.read_csv(r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\metadata_220603\behavior_sessions.csv")

mice =  b['mouse_id'].unique()

H_first_G_mice = [578257, 577287, 570302]

def two_ephys(mdf):
    
    ephys = mdf.loc[(mdf['session_type'].str.contains('EPHYS'))&
                    (~mdf['session_type'].str.contains('pretest'))]
    
    return len(ephys)==2, len(ephys)


def two_ephys_last(mdf):
    
    mdf_sorted = mdf.sort_values('date_of_acquisition')
    return all(mdf_sorted.iloc[-2:]['session_type'].str.contains('EPHYS'))


def two_ephys_different(mdf):
    
    mdf_sorted = mdf.sort_values('date_of_acquisition')
    ephys_sess = mdf_sorted.loc[mdf_sorted['session_type'].str.contains('EPHYS')]
    return len(ephys_sess['session_type'].unique())==2
      

def pre_ephys_same_set(mdf):
    
    pre_ephys_df = mdf.loc[~mdf['session_type'].str.contains('EPHYS')]
    pre_ephys_image_set = pre_ephys_df.image_set.dropna()
    
    return len(np.unique(pre_ephys_image_set)) == 1


def first_ephys_prior_omissions(mdf):
    
    mdf_sorted = mdf.sort_values('date_of_acquisition')
    first_ephys_sess = mdf_sorted.loc[mdf_sorted['session_type'].str.contains('EPHYS')].iloc[0]
    
    return first_ephys_sess['prior_exposures_to_omissions']==0


def first_ephys_right_image_set(mdf):
    
    mdf_sorted = mdf.sort_values('date_of_acquisition')
    training_image_set = mdf.loc[mdf['session_type'].str.contains('TRAINING_3')].iloc[0]['image_set']
    first_ephys_sess = mdf_sorted.loc[mdf_sorted['session_type'].str.contains('EPHYS')].iloc[0]['image_set']
    
    if mdf['mouse_id'].iloc[0] in H_first_G_mice:
        return (training_image_set=='G')&(first_ephys_sess=='H')
    else:
        return training_image_set == first_ephys_sess
    
    
def has_all_training_stages(mdf):
    
    training_stages = mdf.loc[mdf['session_type'].str.contains('TRAINING')]['session_type']
    for stage in ['TRAINING_0', 'TRAINING_1', 'TRAINING_2', 
                  'TRAINING_3', 'TRAINING_4', 'TRAINING_5']:
        if not any(training_stages.str.contains(stage)):
            return False
    
    return True
    
    
validation_funcs = [two_ephys, pre_ephys_same_set, two_ephys_last, two_ephys_different,
                    pre_ephys_same_set, first_ephys_prior_omissions, first_ephys_right_image_set,
                    has_all_training_stages]
validation_dict = {func.__name__:[] for func in validation_funcs}
validation_dict.update({'mouse_id':[]})
for mouse in mice:
    
    mdf = b.loc[b['mouse_id']==mouse]
    mdf = mdf.sort_values('date_of_acquisition')
    validation_dict['mouse_id'].append(mouse)

    for func in validation_funcs:
        validation_dict[func.__name__].append([func(mdf), mouse])

failures = []
for func in validation_dict:
    if func == 'mouse_id':
        continue
    results = validation_dict[func]
    for r in results:
        if r[0]:
            pass
        else:
            failures.append([func, r[1]])



failed_two_ephys_inds = [ind for ind in range(len(validation_dict['two_ephys'])) if validation_dict['two_ephys'][ind][0]==False]
for ind in failed_two_ephys_inds:
    
    mid = validation_dict['mouse_id'][ind]
    print(mid)
    mdf = b.loc[b['mouse_id']==mid]
    mdf = mdf.sort_values('date_of_acquisition')
    print(mdf[['session_type', 'date_of_acquisition']])
    
    print('')
    

#####################
### CHANNEL TABLE ###
def read_json(path):
    
    with open(path, 'r') as f:
        j = json.load(f)
    
    return j

ref_3a = [36, 75, 112, 151, 188, 227, 264, 303, 340, 379]

pj_dir = r'\\allen\programs\braintv\workgroups\neuralcoding\corbettb\VBN_production'

channels = pd.read_csv(r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\metadata_220603\channels.csv")
probes = pd.read_csv(r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\metadata_220603\probes.csv")
cp = channels.merge(probes, left_on='ecephys_probe_id', right_on='ecephys_probe_id')

channel_info = {s:{} for s in cp['ecephys_session_id_x'].unique()}
for sess_id in cp['ecephys_session_id_x'].unique():
    
    c_sess = cp.loc[cp['ecephys_session_id_x']==sess_id]
    
    for pid in c_sess['ecephys_probe_id'].unique():
        
        p_sess = c_sess.loc[c_sess['ecephys_probe_id']==pid]
        probe_name = p_sess['name'].iloc[0]
        probe_json_path = glob.glob(os.path.join(pj_dir, str(sess_id)+'*'+probe_name+'_sorted', 'probe_info.json'))[0]
        pj = read_json(probe_json_path)
        
        total_table_chans = p_sess['valid_data'].sum()
        total_pj_chans = np.sum(pj['mask'])
        
        surface_chan = pj['surface_channel']
        
        top_table_chan = p_sess.loc[p_sess['valid_data']]['probe_channel_number'].max()
        bottom_table_chan = p_sess.loc[p_sess['valid_data']]['probe_channel_number'].min()
        
        pj_mask = np.array(pj['mask'])
        top_pj_chan = np.where(pj_mask)[0].max()
        bottom_pj_chan = np.where(pj_mask)[0].min()
        
        channel_info[sess_id][pid] = {'name': probe_name,
                    'table_chan_num': total_table_chans,
                    'pj_chan_num': total_pj_chans,
                    'surface_chan': surface_chan,
                    'top_table_chan': top_table_chan,
                    'bottom_table_chan': bottom_table_chan,
                    'top_pj_chan': top_pj_chan,
                    'bottom_pj_chan': bottom_pj_chan,
                    'invalid_pj': np.where(pj_mask==False)[0],
                    'invalid_table': p_sess.loc[p_sess['valid_data']==False]['probe_channel_number'].values
                    }
        


pj_num = np.array([sess[p]['pj_chan_num'] for skey, sess in channel_info.items() for p in sess])
table_num = np.array([sess[p]['table_chan_num'] for skey, sess in channel_info.items() for p in sess])
top_table_chan = np.array([sess[p]['top_table_chan'] for skey, sess in channel_info.items() for p in sess])

many_too_few_channels = []
slightly_too_few = []
mask_3a = []
for skey, sess in channel_info.items():
    for pkey, p in sess.items():
        if p['pj_chan_num']<372:
            many_too_few_channels.append((skey, pkey, p['pj_chan_num']))
        elif p['pj_chan_num']==372:
            mask_3a.append((skey, pkey, p['pj_chan_num']))
        elif 383>p['pj_chan_num']>372:
            slightly_too_few.append((skey, pkey, p['pj_chan_num']))

many_too_few_channels = []
slightly_too_few = []
mask_3a = []
for skey, sess in channel_info.items():
    for pkey, p in sess.items():
        if p['pj_chan_num']<372:
            many_too_few_channels.append((skey, pkey, p['pj_chan_num']))
        elif p['pj_chan_num']==372:
            mask_3a.append((skey, pkey, p['pj_chan_num']))
        elif 383>p['pj_chan_num']>372:
            slightly_too_few.append((skey, pkey, p['pj_chan_num']))
        
##########################
### SESSIONS #############
            
 sessions = pd.read_csv(r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\metadata_220603\ecephys_sessions.csv")


def novel_is_novel(sessions):
    
    first_sessions = sessions.loc[sessions['session_number']==1]
    H_first_G = first_sessions.loc[np.isin(first_sessions.mouse_id, H_first_G_mice)]
    rest = first_sessions.loc[~np.isin(first_sessions.mouse_id, H_first_G_mice)]
    
    rest_validation = (rest['prior_exposures_to_image_set']>0) & (rest['experience_level']=='Familiar')
    H_G_validation =  (H_first_G['prior_exposures_to_image_set']==0) & (H_first_G['experience_level']=='Novel')
    failed_sessions = list(rest.iloc[np.where(rest_validation.values==False)[0]]['ecephys_session_id'].values) +\
        list(H_first_G.iloc[np.where(H_G_validation.values==False)[0]]['ecephys_session_id'].values)
    return failed_sessions
    
    
    
def no_prior_is_novel(sessions):
    
    no_prior_sessions = sessions.loc[sessions['prior_exposures_to_image_set']==0]
    validation = no_prior_sessions['experience_level'] == 'Novel'
    failed_sessions = list(no_prior_sessions.iloc[np.where(validation.values==False)[0]]['ecephys_session_id'].values)
    return failed_sessions
    


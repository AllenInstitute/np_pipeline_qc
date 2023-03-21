# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:31:11 2020

@author: svc_ccg
"""
from collections import OrderedDict

import cv2
import h5py
import numpy as np
import pandas as pd
import visual_behavior.analyze

import np_pipeline_qc.legacy.probeSync_qc as probeSync
from np_pipeline_qc.legacy import build_stim_tables, data_getters
from np_pipeline_qc.legacy.query_lims import query_lims
from np_pipeline_qc.legacy.sync_dataset import Dataset as sync_dataset

eye_cam_dict = {'Eye': 'RawEyeTrackingVideo',
                        'Face': 'RawFaceTrackingVideo', 
                        'Side': 'RawBehaviorTrackingVideo'}

class EcephysBehaviorSession():
    '''Get all data from a Visual Behavior Ecephys experiment.
    Can get data from either LIMS or a local data directory
    '''
    
    @classmethod
    def from_lims(cls, ephys_experiment_id: int):
        file_paths = data_getters.lims_data_getter(exp_id=ephys_experiment_id)
        
        return cls(data_paths=file_paths.data_dict)

    
    @classmethod
    def from_local(cls, local_path: str, cortical_sort=False):
        file_paths = data_getters.local_data_getter(base_dir=local_path, cortical_sort=cortical_sort)
        
        return cls(data_paths=file_paths.data_dict)


    @classmethod
    def from_h5(cls, h5path: str):
        
        df_items = ['stim_table', 'unit_table']
        
        hdict = {}
        hdf5_to_dict(hdict, filePath=h5path, grp=None, loadDict=None)
        
        for item in hdict:
            if item in df_items:
                hdict[item] = pd.DataFrame(hdict[item])
                
        return cls(data_paths=hdict['experiment_info'], loaddict=hdict)
        
        
 
    def __init__(self, data_paths=None, loaddict={}):
        
        self.data_paths = data_paths
        
        
        self._sync = None
        self._behavior_data = None
        self._mapping_data = None
        self._replay_data = None
        self._opto_data = None
        self._trials = None
        self._lfp = None
        self._stim_epochs = None
        self._video_data = None
        self._experiment_info = loaddict.get('experiment_info')
        self._frame_times = loaddict.get('frame_times')
        self._reward_times = loaddict.get('reward_times')
        self._unit_table = loaddict.get('unit_table')
        self._stim_table = loaddict.get('stim_table')
        self._lick_times = loaddict.get('lick_times')
        self._running_speed = loaddict.get('running_speed')
        self._cam_frame_times = loaddict.get('cam_frame_times')
        self._opto_stim_table = loaddict.get('opto_stim_table')
        
    
    @property
    def experiment_info(self):
        
        if self._experiment_info is None:
            self._experiment_info = self.data_paths
            self._experiment_info['genotype'] = get_genotype(self.data_paths['es_id'])
            self._experiment_info['monitor_lag'] = probeSync.get_monitor_lag(self.sync)
            
        return self._experiment_info
    
    
    @experiment_info.setter
    def experiment_info(self, value):
        self._experiment_info = value
        
    
    @property
    def sync(self):
        
        if self._sync is None:
            sync_file_path = self.data_paths['sync_file']
            self._sync = sync_dataset(sync_file_path)

        return self._sync
    
    
    @sync.setter
    def sync(self, value):
        self._sync = value
        
    
    @property
    def behavior_data(self):
        
        if self._behavior_data is None:
            self._behavior_data = get_pickle(self.data_paths['behavior_pkl'])
        
        return self._behavior_data
    
   
    @behavior_data.setter
    def behavior_data(self, value):
        self._behavior_data = value


    @property
    def mapping_data(self):
        
        if self._mapping_data is None:
            self._mapping_data = get_pickle(self.data_paths['mapping_pkl'])
        
        return self._mapping_data
    
   
    @mapping_data.setter
    def mapping_data(self, value):
        self._mapping_data = value
        
    
    @property
    def replay_data(self):
        
        if self._replay_data is None:
            self._replay_data = get_pickle(self.data_paths['replay_pkl'])
        
        return self._replay_data
    
   
    @replay_data.setter
    def replay_data(self, value):
        self._replay_data = value
        
    
    @property
    def opto_data(self):
        
        if self._opto_data is None:
            self._opto_data = get_pickle(self.data_paths['opto_pkl'])
            
        return self._opto_data
    
    
    @opto_data.setter
    def opto_data(self, value):
        self._opto_data = value
        
    
#    @property
#    def trials(self) -> pd.DataFrame:
#        """A dataframe containing behavioral trial start/stop times, and trial
#        data of type: pandas.DataFrame"""
#        if self._trials is None:
#            self._trials = self.api.get_trials()
#        return self._trials
#
#    
#    @trials.setter
#    def trials(self, value):
#        self._trials = value
        
    
    @property
    def unit_table(self):

        probes_to_run = self.data_paths['data_probes']
        if self._unit_table is None:
            self._unit_table = probeSync.build_unit_table(probes_to_run, 
                                                          self.data_paths, 
                                                          self.sync)

        return self._unit_table
        
    
    @unit_table.setter
    def unit_table(self, value):
        self._unit_table = value
        
        
    @property
    def stim_table(self):
        
        if self._stim_table is None:
            self._stim_table = build_stim_tables.build_full_NP_behavior_stim_table(
                    self.behavior_data, self.mapping_data, 
                    self.replay_data, self.sync)
        
        return self._stim_table


    @stim_table.setter
    def stim_table(self, value):
        self._stim_table = value
        
    
    @property
    def lfp(self):
        
        lfp_dirs = [self.data_paths['lfp'+pid] for pid in self.data_paths['data_probes']]
        if self._lfp is None:
            self._lfp = probeSync.build_lfp_dict(lfp_dirs, self.sync)
         
        return self._lfp
            
     
    @lfp.setter
    def lfp(self, value):
        self._lfp = value
        
    
    @property
    def video_data(self):
        
        #video_paths = [self.data_paths[eye_cam_dict[vid]] for vid in ['Eye', 'Face', 'Side']]
        if self._video_data is None:
            self._video_data = {}
            for vid in ['Eye', 'Face', 'Side']:
                try:
                    self._video_data[vid] = cv2.VideoCapture(self.data_paths[eye_cam_dict[vid]])
                except Exception as e:
                    print('could not load video {} due to error {}'.format(vid, e))
        
        return self._video_data


    @video_data.setter
    def video_data(self, value):
        self._video_data = value
                     
    
    @property
    def stim_epochs(self):
        
        if self._stim_epochs is None:
            
            behavior_frame_count = self.behavior_data['items']['behavior']['intervalsms'].size + 1
            mapping_frame_count = self.mapping_data['intervalsms'].size + 1
            replay_frame_count = self.replay_data['intervalsms'].size + 1
            
            start_frames = probeSync.get_frame_offsets(self.sync, 
                                                      [behavior_frame_count,
                                                       mapping_frame_count,
                                                       replay_frame_count], 
                                                       tolerance=0)
            end_frames = [start_frames[it]+total-1 for it, total in enumerate([behavior_frame_count,
                                                       mapping_frame_count,
                                                       replay_frame_count])]
            
            self._stim_epochs = {epoch:[start,end] for epoch,start,end in zip(['behavior', 'mapping', 'replay'],
                                                                             start_frames, end_frames)}   
    
        return self._stim_epochs
    

    @stim_epochs.setter
    def stim_epochs(self, value):
        self._stim_epochs = value
        
    
    @property
    def frame_times(self):
        
        if self._frame_times is None:
            self._frame_times = probeSync.get_vsyncs(self.sync, fallback_line=2)
        
        return self._frame_times
    

    @frame_times.setter
    def frame_times(self, value):
        self._frame_times = value
        
    
    @property
    def running_speed(self):
        
        if self._running_speed is None:
            running_speed = np.concatenate([get_running_from_pkl(pkl) for pkl in 
                               [self.behavior_data, self.mapping_data, self.replay_data]])
    
            running_time = self.frame_times
            
            self._running_speed = [running_speed, running_time]
            
        return self._running_speed

    
    @running_speed.setter
    def running_speed(self, value):
        self._running_speed = value
    
        
    @property
    def cam_frame_times(self):
        
        eye_cam_dict = {'Eye': 'RawEyeTrackingVideo',
                        'Face': 'RawFaceTrackingVideo', 
                        'Side': 'RawBehaviorTrackingVideo'}
        
        
        if self._cam_frame_times is None:
            self._cam_frame_times = {}
            for cam in eye_cam_dict:
                cam_json = self.data_paths.get(eye_cam_dict[cam] + 'Metadata')
                if cam_json:
                    cam_frame_times = probeSync.get_frame_exposure_times(self.sync, cam_json)
                    self._cam_frame_times[cam] = cam_frame_times
        return self._cam_frame_times
    

    @cam_frame_times.setter
    def cam_frame_times(self, value):
        self._cam_frame_times = value


    @property
    def lick_times(self):
        
        if self._lick_times is None:
            self._lick_times = probeSync.get_lick_times(self.sync)
        
        return self._lick_times
    

    @lick_times.setter
    def lick_times(self, value):
        self._lick_times = value


    @property
    def reward_times(self):
        
        if self._reward_times is None:
            reward_frames = self.behavior_data['items']['behavior']['rewards'][0]['reward_times'][:, 1]
            self._reward_times = self.frame_times[reward_frames.astype(int)]
            
        return self._reward_times

    @reward_times.setter
    def reward_times(self, value):
        self._reward_times = value
                   
    
    @property
    def opto_stim_table(self):
        
        if self._opto_stim_table is None:
            self._opto_stim_table = build_stim_tables.get_opto_stim_table(self.sync, self.opto_data)

        return self._opto_stim_table

    @opto_stim_table.setter
    def opto_stim_table(self, value):
        self._opto_stim_table = value
        

def get_running_from_pkl(pkl):
    
    key = 'behavior' if 'behavior' in pkl['items'] else 'foraging'
    intervals = pkl['items']['behavior']['intervalsms'] if 'intervalsms' not in pkl else pkl['intervalsms']
    time = np.insert(np.cumsum(intervals), 0, 0)/1000.
    
    dx,vsig,vin = [pkl['items'][key]['encoders'][0][rkey] for rkey in ('dx','vsig','vin')]
    run_speed = visual_behavior.analyze.compute_running_speed(dx[:len(time)],time,vsig[:len(time)],vin[:len(time)])
    return run_speed


def get_pickle(pickle_path):
    return pd.read_pickle(pickle_path)


def save_to_h5(obj, savepath, attributes_to_save = ['unit_table',
                          'stim_table',
                          'opto_stim_table',
                          'running_speed',
                          'reward_times',
                          'lick_times',
                          'cam_frame_times',
                          'experiment_info',
                          'frame_times']):
    
    
    with h5py.File(savepath,'a') as savefile:
        
        for a in attributes_to_save:
            grp = savefile['/']
            attr = getattr(obj, a)
            if isinstance(attr, pd.DataFrame):
                attr = attr.to_dict()
            
            if not isinstance(attr, dict):
                attr = {a: attr}
            
            add_to_hdf5(savefile, grp=grp.create_group(a), saveDict=attr)
        
        savefile.close()
        
        
def add_to_hdf5(savefile, grp=None, saveDict=None):
    
    if grp is None:    
        grp = savefile['/']
        
    for key in saveDict:
        
        if isinstance(saveDict[key],(dict,OrderedDict)):
            print(grp.name)
            subgrp = grp.create_group(str(key))
            add_to_hdf5(savefile, grp=subgrp, saveDict=saveDict[key])
        elif isinstance(saveDict[key], pd.DataFrame):
            print(grp.name)
            subgrp = grp.create_group(str(key))
            add_to_hdf5(savefile, grp=subgrp, saveDict=saveDict[key].to_dict())
        else:
            try:
                grp.create_dataset(str(key),data=saveDict[key],compression='gzip',compression_opts=1)
            except:
                try:
                    grp[str(key)] = saveDict[key]
                except:
                    #try:
                    grp.create_dataset(str(key),data=np.array(saveDict[key],dtype=object),dtype=h5py.special_dtype(vlen=str))
#                    except:
#                        print('Could not save: ', key)


def hdf5_to_dict(parentdict, filePath=None, grp=None, loadDict=None):
    if grp is None:
        grp = h5py.File(filePath,'r')
        newFile = grp
    else:
        newFile = None
    for key,val in grp.items():
        if isinstance(val,h5py._hl.dataset.Dataset):
            v = val[()]
            if isinstance(v,np.ndarray) and v.dtype==np.object:
                v = v.astype('U')
            if loadDict is None:
                #setattr(obj,key,v)
                parentdict[key] = v
            else:
                loadDict[key] = v
        elif isinstance(val,h5py._hl.group.Group):
            if loadDict is None:
                #setattr(obj,key,{})
                parentdict[key] = {}
                hdf5_to_dict(parentdict,grp=val,loadDict=parentdict[key])
            else:
                loadDict[key] = {}
                hdf5_to_dict(parentdict,grp=val,loadDict=loadDict[key])
    if newFile is not None:
        newFile.close()
                

def get_genotype(es_id):
    query_string = '''
        SELECT es.id as es_id, sp.name as specimen_name
        FROM ecephys_sessions es
        JOIN specimens sp ON sp.id = es.specimen_id
        WHERE es.id = {}
        ORDER BY es.id
        '''
    try:
        genotype_info = query_lims(query_string.format(es_id))
        if len(genotype_info)>0 and 'specimen_name' in genotype_info[0]:
            genotype_string = genotype_info[0]['specimen_name']
            genotype = genotype_string[:genotype_string.rfind('-')]
        else:
            print('Could not find genotype for session {}'.format(es_id))
            genotype = ''
    except Exception as e:
        genotype = ''
        print('Error retrieving genotype: {}'.format(e))
    
    return genotype




    

import os
import json
import pprint
#from recordclass import recordclass
from collections import namedtuple, OrderedDict 
import glob
import logging
import shutil
import pandas as pd


logging.basicConfig(level=logging.WARNING)
save_dir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\sorted_data_validation_results"
source_volume_config = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\source_list.json"

def get_path(path_in):
    try:
        path_out = glob.glob(path_in)
        if len(path_out) == 1:
            return(path_out[0])
        else:
            logging.info(f'ambiguous paths: {path_out}')
    except Exception as E:
        logging.info('path check failed')

class ecephys_files():
    def __init__(self, path, network_drive = r"\\sd4\SD4", acq_computer_name=None):
        self.path = path
        self.session = os.path.split(path)[1]
        self.network_drive = network_drive
        self.network_dir = get_network_dir(path)
        print('Session name: {}'.format(self.session))
        print('Network directory: {}'.format(self.network_dir))


        self.inserted_probes = get_inserted_probes_from_platformD1json(self.network_dir)
        print('Inserted probes: {}'.format(self.inserted_probes))

        logging.basicConfig(level = logging.INFO)
        self.logger = logging.getLogger()

        self.extension_params = namedtuple('extension_params', ['exp_only', 'size_min_rel', 'size_min_abs'])

#    self.named_file_state = recordclass('file_state',['local','platform_json','network','size_match'])
#    self.file_state_dict = {}
#    for extension in self.file_extensions:
#           self.file_state_dict[extension] = self.named_file_state(False,False,False,False)

        self.named_daq = namedtuple('named_daq', ['computername','backups'])

        if acq_computer_name is None:
            print('Please specify acquisition computer name')
            # self.daqs = {
            #                         'daq1':self.named_daq(config['components']['OpenEphys']['host'],['E']),
            #                         #'daq2':self.named_daq('W10DT05501',['N','M','I','H','D','E','F','G'])                
            #                         }
        else:
            self.daqs = {
                                    'daq1':self.named_daq(acq_computer_name,['E']),
                                    }
        
        self.named_probe = namedtuple('named_probe',['daq','drive','size_min_rel','size_min_abs'])

                
        #self.pull_json()
        self.hab = False
        self.network_path =  os.path.join(self.network_drive,self.session)
        self.data = {
             'probeA':self.named_probe('daq1','E',0,0),#last two params are rel size and abs size mininmums
             'probeB':self.named_probe('daq1','E',0,0),#use self.eye_vid_size to determing these, should be good to go
             'probeC':self.named_probe('daq1','E',0,0),
             'probeD':self.named_probe('daq1','E',0,0),
             'probeE':self.named_probe('daq1','E',0,0),
             'probeF':self.named_probe('daq1','E',0,0)
             }

        #filter out probes that weren't inserted
        self.data = {p:self.data[p] for p in self.inserted_probes}


        self.file_extension_dict = self.create_file_extensions_dict()
        self.relpaths,self.data_files = self.make_files()


        self.check_dict = OrderedDict()
        self.check_params = namedtuple('check_params',['iterable','function','true_string','false_string'])
        self.true_dict = OrderedDict()
        self.false_dict = OrderedDict()
        self.size_dict = {
            'data':{},
            'data_sorted':{},
            'extension':{},
            'extension_network':{},
            'data_network':{},
            'data_sorted_network':{},
            'data_files':{},
            'data_files_network':{},
            'data_sorted_files':{},
            'data_sorted_files_network':{}
        }
 

        #self.add_check_params('data_sorted', self.check_params(self.data,self.check_data_sorted,
        #        'All the sorted data directories exist on the acquisition drive and are named properly!',
        #        'sorted data directories are missing from the acquisition drive or improperly named:' ))

        #self.add_check_params('data_sorted_backup', self.check_params(self.data,self.check_data_sorted_backup,
        #    'All the sorted data directories exist on the backup drives and are named properly!',
        #    'sorted data directories are missing from the backup drives or improperly named:' )) 

        #self.add_check_params('data_listed_extra', self.check_params(self.platform_json_dirs,self.check_data_dir_extra,
        #     'All data directories listed in the platformD1_json are present!',
        #    'data directories in the platformD1_json don\'t exist:' ))

        self.add_check_params('lims_files', self.check_params(self.data,self.check_lims2_files,
                 'All lims2 files are present!',
                'Some Lims2 files are missing:' ))


    def pull_json(self):
            json_path = os.path.join(self.path,self.session+'_platformD1.json')
            with open(json_path, 'r') as platform_json_file:
                platform_json = json.load(platform_json_file)
                #pprint.plogging.info(platform_json['files'])
                self.platform_json_files = []
                self.platform_json_dirs = []
                for value in platform_json['files'].values():
                        try:
                                self.platform_json_files.append(value['filename'])
                        except KeyError:
                                self.platform_json_dirs.append(value['directory_name'])
                workflow_dict = {
                                'manual_habituation':True,
                'manual_experiment':False,
                'neuropixel.neuropixel2_workflow':False,
                'neuropixel.neuropixel1_workflow':False,
                'neuropixel.neuropixel_workflow':False,
                'neuropixel.habituation_workflow':True
                }
                self.hab = workflow_dict[platform_json['workflow']]
                
                if not(self.hab):
                        self.hab = False
                        self.network_path =  os.path.join(self.network_drive,self.session)
                        self.data = {
                             'probeA':self.named_probe('daq1','H',0,0),#last two params are rel size and abs size mininmums
                             'probeB':self.named_probe('daq1','K',0,0),#use self.eye_vid_size to determing these, should be good to go
                             'probeC':self.named_probe('daq1','L',0,0),
                             'probeD':self.named_probe('daq2','J',0,0),
                             'probeE':self.named_probe('daq2','K',0,0),
                             'probeF':self.named_probe('daq2','L',0,0)
                             }
                             
                if self.hab:
                    self.data = {}
                    session_type_dict = {
                    'Exp':'',
                    'Hab1':'Habituation_day1',
                    'Hab2':'Habituation_day2'
                    }
                    rig_session_dict = {
                    'NP.1':'Habituation_day2',
                    'NP.2':'Habituation_day1',
                    'NP.0':'Habituation_day1'
                 }
                    try:
                        surgery_json_path = os.path.join(self.path,self.session+'_surgeryNotes.json')
                        with open(surgery_json_path, 'r') as surgery_json_file:
                                surgery_notes = json.load(surgery_json_file)
                        SessionType = surgery_notes['session_notes']['SessionType']
                        self.network_path =  os.path.join(self.network_drive,session_type_dict[SessionType],self.session)
                    except KeyError as E:
                        Rig = platform_json['rig_id']
                        self.network_path =  os.path.join(self.network_drive,rig_session_dict[Rig],self.session)




    def create_file_extensions_dict(self):
        possible_extensions_dict = {
            ".behavior.avi": self.extension_params(False, 999, 999),
            ".eye.avi": self.extension_params(False, 999, 999),
            ".stim.pkl": self.extension_params(False, 999, 999),
            "_report.pdf": self.extension_params(False, 999, 999),
            "_surface-image1-left.png": self.extension_params(False, 999, 999),
            "_surface-image1-right.png": self.extension_params(False, 999, 999),
            "_surgeryNotes.json": self.extension_params(False, 999, 999),
            ".sync": self.extension_params(False, 999, 999),
            "_platformD1.json": self.extension_params(False, 999, 999),
            ".motor-locs.csv": self.extension_params(True, 999, 999),
            ".opto.pkl": self.extension_params(True, 999, 999),
            "_surface-image2-left.png": self.extension_params(True, 999, 999),
            "_surface-image2-right.png": self.extension_params(True, 999, 999),
            "_surface-image3-left.png": self.extension_params(True, 999, 999),
            "_surface-image3-right.png": self.extension_params(True, 999, 999),
            "_surface-image4-left.png": self.extension_params(True, 999, 999),
            "_surface-image4-right.png": self.extension_params(True, 999, 999),
            "_surface-image5-left.png": self.extension_params(True, 999, 999),
            "_surface-image5-right.png": self.extension_params(True, 999, 999),
            "_surface-image6-left.png": self.extension_params(True, 999, 999),
            "_surface-image6-right.png": self.extension_params(True, 999, 999)
            }
        file_extension_dict = {}
        for extension,params in possible_extensions_dict.items():
            if self.hab and params.exp_only: pass
            else: 
                file_extension_dict[extension] = params
        return file_extension_dict

    def make_files(self):
        data_file_params = namedtuple('data_file_params',['relpath','upload','sorting_step'])
        relpaths = {
                                        'lfp':r"continuous\Neuropix-PXI-100.1",
                                        'spikes':r"continuous\Neuropix-PXI-100.0",
                                        'events':r"events\Neuropix-PXI-100.0\TTL_1",
                                        'empty':""
                                                }
                        
        data_files = {
                    "probe_info.json":data_file_params('empty',True,'depth_estimation'),
                    "channel_states.npy":data_file_params('events',True,'extraction'),
                    "event_timestamps.npy":data_file_params('events',True,'extraction'),
                    r"continuous\Neuropix-PXI-100.1\continuous.dat":data_file_params('empty',True,'extraction'),
                    "lfp_timestamps.npy":data_file_params('lfp',True,'sorting'),
                    "amplitudes.npy":data_file_params('spikes',True,'sorting'),
                    "spike_times.npy":data_file_params('spikes',True,'sorting'),
                            "mean_waveforms.npy":data_file_params('spikes',True,'mean waveforms'),
                            "spike_clusters.npy":data_file_params('spikes',True,'sorting'),
                            "spike_templates.npy":data_file_params('spikes',True,'sorting'),
                            "templates.npy":data_file_params('spikes',True,'sorting'),
                            "whitening_mat.npy":data_file_params('spikes',True,'sorting'),
                            "whitening_mat_inv.npy":data_file_params('spikes',True,'sorting'),
                            "templates_ind.npy":data_file_params('spikes',True,'sorting'),
                            "similar_templates.npy":data_file_params('spikes',True,'sorting'),
                            "metrics.csv":data_file_params('spikes',True,'metrics'),
                            "channel_positions.npy":data_file_params('spikes',True,'sorting'),
                            "cluster_group.tsv":data_file_params('spikes',False,'sorting'),
                            "channel_map.npy":data_file_params('spikes',True,'sorting'),
                            "params.py":data_file_params('spikes',True,'sorting'),
                    "probe_depth_*.png":data_file_params("empty",True,'depth estimation'),
                    r"continuous\Neuropix-PXI-100.0\continuous.dat":data_file_params('empty',False,'extraction'),
                    "residuals.dat":data_file_params('spikes',False,'median subtraction'),
                    "pc_features.npy":data_file_params('spikes',False,'sorting'),
                    "template_features.npy":data_file_params('spikes',False,'sorting'),
                    "rez2.mat":data_file_params('spikes',False,'sorting'),
                    "rez.mat":data_file_params('spikes',False,'sorting'),
                    "pc_feature_ind.npy":data_file_params('spikes',False,'sorting'),
                    "template_feature_ind.npy":data_file_params('spikes',False,'sorting')
                    }
        return relpaths, data_files

    def add_check_params(self, key, check_params):
        self.check_dict[key] = check_params
        self.check_thing(key,check_params)

    def check_thing(self,key,params):
        true_list = []
        false_list = []
        for thing in params.iterable:
            failure_dict = params.function(thing)
            if failure_dict:
                logging.warning(thing + 'failed!!!')
                logging.warning('the following files had errors, see the json for more details:')
                logging.warning(failure_dict.keys())
                false_list.append(thing)
            else: 
                true_list.append(thing)
                print(thing + 'Passed!!!')
            save_name =self.session +'_'+ thing+'_sorted_data_lims_upload_validation_results.json'
            #save_dir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\sorted_data_validation_results"
            save_path = os.path.join(save_dir, save_name)
            os.makedirs(save_dir, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(failure_dict, f, indent=2)
        if true_list:
            if false_list:
                logging.warning("ERROR - Some "+params.false_string)
                for thing in false_list:
                        logging.info("     " +thing)
            else:
                print(params.true_string)
        elif false_list:
                logging.warning("ERROR - All "+params.false_string)
        else: logging.warning("Nothing to check:. Did not check if " + params.false_string)        
        self.true_dict[key], self.false_dict[key] = true_list,false_list

        
    
    def dir_size(self, dir_path, key, probe):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(dir_path):
                    for f in filenames:
                            fp = os.path.join(dirpath, f)
                            fsize = os.path.getsize(fp)
                            total_size +=fsize
                            #self.size_dict[key][probe][f] = fsize
            return total_size


    def check_data_sorted(self,probe):
        daq = self.data[probe].daq
        computer_name = self.daqs[daq].computername
        data_path = os.path.join(r"\\"+computer_name, self.data[probe].drive,self.session+"_"+probe+"_sorted")
        size = False
        if os.path.isdir(data_path):
            self.size_dict['data_sorted_files'][probe] = {}
            size = self.dir_size(data_path,'data_sorted_files',probe)
        self.size_dict['data_sorted'][probe] = size
        return bool(size)

    #check data sorted ready for upload?

    def check_data_sorted_backup(self,probe):
        daq = self.data[probe].daq
        computer_name = self.daqs[daq].computername
        #TODO restructure so that you use try instead of ifs (just try getsize directly))
        size = False
        for drive in self.daqs[daq].backups:
            if not size:
                try:
                    try_data_path = os.path.join(r"\\"+computer_name, drive,self.session+"_"+probe+"_sorted")
                    size = self.dir_size(try_data_path,'data_sorted_files',probe)
                except Exception as E:
                    #pass
                    logging.info("Catch the specific exception now that you know what it is")
                    raise E
        self.size_dict['data_sorted'][probe] = size
        return bool(size)

    def check_data_sorted_backup_size(self,probe):
        min_size = (self.size_dict['data'][probe]*1.4)+(10*(10**9))
        return min_size < self.size_dict['data_sorted'][probe]
    
    def check_data_dir_extra(self,dirname):
        probe = dirname.split("_")[-1]
        return probe in self.true_dict['data_sorted']

    def check_quality_column(self, metrics_path):
        metrics_csv = pd.read_csv(metrics_path)
        quality_column_present = False
        if 'quality' in metrics_csv.columns:
            quality_column_present = True
            #print('found quality')
        else:
            pass#logging.info('Failed to find the quality column in :'+metrics_path)
        return quality_column_present


    def check_probe_info_json(self, probe_info_path):
        with open(probe_info_path, 'r') as f:
            probe_info = json.load(f)
        probe_serial_number_found = False
        if 'probe' in probe_info:
            probe_serial_number_found = True#probe_info['probe']
        else:
            pass#logging.info('Failed to find the probe serial number in :'+probe_info_path)
        return probe_serial_number_found

    def check_probe_depth_png_name(self, probe_info_path, sorted_probe):
        name_correct = False
        last_char = os.path.splitext(probe_info_path)[0][-1]
        print(last_char)
        print('should match: ', sorted_probe)
        if sorted_probe[-1] == last_char:
            name_correct = True
        else:
            pass#logging.info('Failed to find the probe letter in the png:'+probe_info_path)
        return name_correct

    def check_not_cortical_sort(self, metrics_path, computer_name,acq_drive,session_probe):
        not_cortical_sort = False
        try:
            file = 'probe_info.json'
            json_params = self.data_files[file]
            relpath = self.relpaths[json_params.relpath]
            probe_json_path = get_path(os.path.join(r"\\",computer_name,acq_drive,session_probe,relpath,file))
            with open(probe_json_path, 'r') as f:
                probe_info = json.load(f)
            surface_channel = probe_info['surface_channel']
            metrics_csv = pd.read_csv(metrics_path)
            peak_chans = metrics_csv.loc[:, 'peak_channel']
            low_chan = min(peak_chans)
            if int(low_chan) < int(surface_channel)-85: #(should this be min of 0? 5? if surface chan is like 80)
                not_cortical_sort = True
        except Exception as E:
            logging.error('checking cortical sort failed', exc_info=True)
        return not_cortical_sort

    def screen_for_cortical_sort(self, metrics_path, channel_range_thresh=150):
         metrics_csv = pd.read_csv(metrics_path)
         peak_channels = metrics_csv['peak_channel']
         channel_range = peak_channels.max() - peak_channels.min()
         print('Peak Channel min: {}    max: {}'.format(peak_channels.min(), peak_channels.max()))
         if channel_range<channel_range_thresh:
             print('Peak channel range <{} in metrics file. Suspected cortical sort.'.format(channel_range_thresh))
         return channel_range>=channel_range_thresh

    def check_timestamps_not_corrupted(self, timestamps_path):
        not_corrupted = False
        if os.path.getsize(timestamps_path)>40000:#This is 40kb. might help check for truncated sorts too
            not_corrupted = True
        return not_corrupted
#TODO 
    def check_lims2_files(self,sorted_probe):
        print('')
        session_probe = self.session+'_'+sorted_probe+'_sorted'
        daq = self.data[sorted_probe].daq
        computer_name = self.daqs[daq].computername
        acq_drive = self.data[sorted_probe].drive
        def check_lims_files(self,computer_name,acq_drive,session_probe, sorted_probe):
            failed_files_dict = {}
            test_list = ['quality_column', 'full_probe_sort1', 'full_probe_sort2', 'probe_info_in_probe_json', 'probe_letter_in_png_filename', 
                'timestamps_not_corrupted']# add later - , 'post_MS_noise', 'full_extraction'
            
            for file,file_params in self.data_files.items():
                relpath = self.relpaths[file_params.relpath]   
                local_path = get_path(os.path.join(r"\\",computer_name,acq_drive,session_probe,relpath,file))
                print(local_path)
                found = False
                upload = file_params.upload
                #logging.info(local_path)
                if local_path:
                    try:
                        #logging.info('looking for :'+local_path)
                        
                        found = bool(os.path.isfile(local_path))
                        if found and file == 'metrics.csv':
                            #print('testing quality')
                            test_list.remove('quality_column')
                            #print(test_list)
                            test = self.check_quality_column(local_path)
                            #print(test)
                            if not(test==True):
                                failed_files_dict[file+'_qulity'] = 'Failed to find the quality column' 
                            #print()

                        if found and file == 'metrics.csv':
                            test_list.remove('full_probe_sort1')
                            test = self.screen_for_cortical_sort(local_path)
                            if not(test==True):
                                failed_files_dict[file+'_cortical_sort1'] = 'Looks like this might be a cortical_sort' 
                            #logging.info('found metrics column')

                        if found and file == 'metrics.csv':
                            test_list.remove('full_probe_sort2')
                            test = self.check_not_cortical_sort(local_path, computer_name,acq_drive,session_probe)
                            if not(test==True):
                                failed_files_dict[file+'_cortical_sort2'] = 'Looks like this might be a cortical_sort' 
                            #logging.info('found metrics column')

                        if found and file == "probe_info.json":
                            test_list.remove('probe_info_in_probe_json')
                            test = self.check_probe_info_json(local_path)
                            if not(test==True):
                                failed_files_dict[file] = 'Failed to find the probe serial number and other "probe" info' 
                        
                        if found and file == "probe_depth_*.png":
                            test_list.remove('probe_letter_in_png_filename')
                            test = self.check_probe_depth_png_name(local_path, sorted_probe)
                            if not(test == True):
                                failed_files_dict[file] = 'Failed to find the probe letter in the filename'                      

                        if found and file == "event_timestamps.npy":
                            test_list.remove('timestamps_not_corrupted')
                            test = self.check_timestamps_not_corrupted(local_path)
                            if not(test == True):
                                failed_files_dict[file] = 'Event timestamps is too small. It may be corrupted or the data was only partially extracted'                      

                    except Exception as E:
                        logging.error('An error occurred', exc_info=True)
                #print('thet test list here: '+str(test_list))
                if found and not(upload):
                    failed_files_dict[file] = 'Non-upload file found'
                if not(found) and upload:
                    failed_files_dict[file] = 'Upload file not found'
            for test in test_list:
                failed_files_dict[test] = 'Failed to perform test'
            if failed_files_dict:
                pass    #self.logger.error(missing_files_list)
            return failed_files_dict

        failed_files_dict = check_lims_files(self,computer_name,acq_drive,session_probe, sorted_probe)
        return failed_files_dict
"""
        if missing_files_list:
            self.logger.warning('Attempting to retrive files not found on acquisition drive for '+sorted_probe)
            logging.info(missing_files_list)

            for file in missing_files_list:
                moved = False
                for drive in self.daqs[daq].backups:
                    if not moved:
                        try:
                            relpath = self.relpaths[self.data_files[file].relpath]
                            try_file_path = os.path.join(r"\\"+computer_name, drive,session_probe,relpath,file)
                            logging.info(try_file_path)
                            local_path = os.path.join(r"\\",computer_name,acq_drive,session_probe,relpath,file)
                            shutil.copyfile(try_file_path,local_path)
                            moved = True
                        except (FileNotFoundError,PermissionError) as E:
                            self.logger.info(E,exc_info = True)

            missing_files_list = check_lims_files(self,computer_name,acq_drive,session_probe)
"""
def get_network_dir(path):
    # vols_to_check = [
    #                 r'\\10.128.50.43\sd6.3',
    #                 r'\\10.128.50.20\sd7',
    #                 r'\\10.128.50.20\sd7.2',
    #                 r'\\10.128.54.20\sd8',
    #                 r'\\10.128.54.20\sd8.2',
    #                 r'\\10.128.54.20\sd8.3',
    #                 r"\\10.128.54.19\sd9"
    #                 ]
    with open(source_volume_config, 'r') as f:
        vols_to_check = json.load(f)
        
    for vol in vols_to_check:

        dirs_in_vol = os.listdir(vol)
        matching_dir = [d for d in dirs_in_vol if path==d]
        if len(matching_dir)>0 and os.path.isdir(os.path.join(vol, matching_dir[0])):
            return os.path.join(vol, matching_dir[0])


def get_inserted_probes_from_platformD1json(network_dir):

    probe_keys = ['ProbeA', 'ProbeB', 'ProbeC', 'ProbeD', 'ProbeE', 'ProbeF']
    probes_inserted = []
    platform_json_file = glob.glob(os.path.join(network_dir, '*platformD1.json'))
    #print(platform_json_file)
    if len(platform_json_file)>0:
        with open(platform_json_file[0], 'r') as f:
            pj = json.load(f)
        #print(pj)
        insertion_notes = pj['InsertionNotes']
        #print('printing insertion notes')
        #print(insertion_notes)
        for p in probe_keys:
            if p in insertion_notes:
                if insertion_notes[p]['FailedToInsert'] == 0:
                    probes_inserted.append(p)
                else:
                    print('{} failed to insert, ignoring'.format(p))
            else:
                #assume that no notes means probe was inserted
                probes_inserted.append(p)
    else:
        #if you can't find platform json, assume all probes inserted
        probes_inserted = probe_keys

    probes_inserted = ['p'+p[1:] for p in probes_inserted] #switch to lowercase p
    #print(probes_inserted)
    #raise(ValueError)
    return probes_inserted


def run_session(sessionID, acq_computer_name):
    session_instance = ecephys_files(sessionID, acq_computer_name=acq_computer_name)
    failed_probes = []
    for probe in session_instance.inserted_probes:
        #print(session_instance.inserted_probes)
        save_name = sessionID +'_'+ probe+'_sorted_data_lims_upload_validation_results.json'
        #save_dir = r'C:\Users\svc_neuropix\Documents\python_scripts\sorted_data_validation_results'
        save_path = os.path.join(save_dir, save_name)
        with open(save_path, 'r') as f:
            failures = json.load(f)
        if failures:
            failed_probes.append(probe)
        else:
            
    if failed_probes == []:
        print(sessionID + ' Passed!!')
        return True
    else:
        print(sessionID + ' Failed. Check the json for the following probes '+', '.join(failed_probes))
        return False


if __name__ == '__main__':
    session_list = [
    "1122903357_570302_20210818",
    "1120251466_578003_20210805",
    "1123100019_570302_20210819",
    "1119946360_578003_20210804",
    "1118512505_576324_20210729",
    "1115356973_570299_20210714",
    "1115077618_570299_20210713",
    "1113954991_578002_20210708",
    "1109680280_568022_20210616",
    "1109889304_568022_20210617",
    "1111013640_568963_20210623",
    "1111216934_568963_20210624",
    "1113957627_562033_20210708",
    "1113751921_562033_20210707",
    "1112515874_572846_20210701",
    ]


    class_dict = {}
    for session in session_list:
        class_dict[session] = ecephys_files(session)
        print("\n\n\n")

    for session, session_instance in class_dict.items():
        failed_probes = []
        for probe in session_instance.inserted_probes:
            #print(session_instance.inserted_probes)
            save_name =session +'_'+ probe+'_sorted_data_lims_upload_validation_results.json'
            #save_dir = r'C:\Users\svc_neuropix\Documents\python_scripts\sorted_data_validation_results'
            save_path = os.path.join(save_dir, save_name)
            with open(save_path, 'r') as f:
                failures = json.load(f)
            if failures:
                failed_probes.append(probe)
        if failed_probes == []:
            print(session + ' Passed!!')
        else:
            print(session + ' Failed. Check the json for the following probes '+', '.join(failed_probes))








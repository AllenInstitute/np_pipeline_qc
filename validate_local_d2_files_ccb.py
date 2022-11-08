import os
import json
from collections import namedtuple, OrderedDict 
import glob
import pandas as pd
import get_sessions as gs
import numpy as np

# This specifies where to look for our network backups
source_volume_config = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\source_list.json"
with open(source_volume_config, 'r') as f:
    sources = json.load(f)

# configuring how to find paths to the sorting data
default_sorting_directory = 'E' 
default_probe_directory_format = r"{}_{}_sorted"
data_file_params = namedtuple('data_file_params',['relpath','upload','sorting_step'])
relpaths = {
                                'lfp':r"continuous\Neuropix-PXI-100.1",
                                'spikes':r"continuous\Neuropix-PXI-100.0",
                                'events':r"events\Neuropix-PXI-100.0\TTL_1",
                                'empty':""
                                        }
                
data_files = { 
        'probe_info.json': data_file_params(relpath='empty', upload=True, sorting_step='depth_estimation'),
        'channel_states.npy': data_file_params(relpath='events', upload=True, sorting_step='extraction'),
        'event_timestamps.npy': data_file_params(relpath='events', upload=True, sorting_step='extraction'),
        'continuous.dat': data_file_params(relpath='lfp', upload=True, sorting_step='extraction'),
        'lfp_timestamps.npy': data_file_params(relpath='lfp', upload=True, sorting_step='sorting'),
        'amplitudes.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'spike_times.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'mean_waveforms.npy': data_file_params(relpath='spikes', upload=True, sorting_step='mean waveforms'),
        'spike_clusters.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'spike_templates.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'templates.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'whitening_mat.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'whitening_mat_inv.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'templates_ind.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'similar_templates.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'metrics.csv': data_file_params(relpath='spikes', upload=True, sorting_step='metrics'),
        'channel_positions.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
#        'cluster_group.tsv': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'channel_map.npy': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'params.py': data_file_params(relpath='spikes', upload=True, sorting_step='sorting'),
        'probe_depth.png': data_file_params(relpath='empty', upload=True, sorting_step='depth estimation'),
        }


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


def validate_d2_files(sessionID, acq_computer_name):
    ''' Main entry point, runs a bunch of validation functions
        on the spike sorting data to check it for readiness to upload
        to lims'''
    
    acq_computer_name = r'\\'+acq_computer_name
    failures = []

    #Get network session directory for this session
    network_session_directory = gs.get_sessions(sources, limsID=sessionID)
    if len(network_session_directory)>0:
        network_session_directory = network_session_directory[0]
    else:
        failures.append(('all probes', 'Could not find session directory: {}'.format(network_session_directory)))
        return failures
    
    print('Found network directory {}'.format(network_session_directory))
    probes_to_run = get_inserted_probes_from_platformD1json(network_session_directory)
    print('running validation on following probes: {}'.format(probes_to_run))
    
    for probe in probes_to_run:
        expected_probe_base = os.path.join(acq_computer_name, 
                                            default_sorting_directory,
                                            default_probe_directory_format.format(sessionID, probe))
        
        if not os.path.exists(expected_probe_base):
            expected_probe_base = os.path.join(r"\\allen\programs\mindscope\workgroups\np-exp",
                                               sessionID,default_probe_directory_format.format(sessionID, probe))
        
        
        probe_base = glob.glob(expected_probe_base)
        print(os.path.normpath(expected_probe_base))
        
        #Check sorting data directory exists for this probe
        if len(probe_base)==0:
            failures.append((probe, 'Sorting data did not exist'))
            continue
        else:
            probe_base = probe_base[0]

        #If the sorting directory exists, check that it has the expected files
        probe_files = {}
        missing_files = []
        for sorting_file_name, sorting_file in data_files.items():
            if sorting_file_name == 'probe_depth.png':
                sorting_file_name = 'probe_depth_{}.png'.format(probe[-1])
            expected_path = os.path.join(probe_base,
                                        relpaths[sorting_file.relpath],
                                        sorting_file_name)

            file_path = glob.glob(expected_path)
            if len(file_path)>0:
                probe_files[sorting_file_name] = file_path[0]
            
            else:
                missing_files.append(expected_path)
                
            if len(missing_files)>0:
                for m in missing_files:
                    failures.append((probe, 'Could not find {}'.format(m)))
                continue

        #If it has the expected files, run validation functions
        check_quality_column(probe, probe_files, failures)
        screen_for_cortical_sort(probe, probe_files, failures)
        check_probe_info_json(probe, probe_files, failures)
        check_npy_data_lengths(probe, probe_files, failures)

    for f in failures:
        print(f)

    return len(failures)==0


def check_quality_column(probe, probe_files, failures):
    '''Make sure the metrics files has the quality 
        column that gets added by the quality metrics module'''
    metrics_path = probe_files['metrics.csv']
    metrics_csv = pd.read_csv(metrics_path)
    if 'quality' in metrics_csv.columns:
        pass
    else:
        failures.append((probe, 'No quality column in metrics csv file'))


def screen_for_cortical_sort(probe, probe_files, failures, channel_range_thresh=150):
    metrics_path = probe_files['metrics.csv']
    metrics_csv = pd.read_csv(metrics_path)
    peak_channels = metrics_csv['peak_channel']
    channel_range = peak_channels.max() - peak_channels.min()
    print('Peak Channel min: {}    max: {}'.format(peak_channels.min(), peak_channels.max()))
    if channel_range<channel_range_thresh:
        failures.append((probe, 'Peak channel range <{} in metrics file. Suspected cortical sort.'.format(channel_range_thresh)))


def check_probe_info_json(probe, probe_files, failures):
    probe_info_path = probe_files['probe_info.json']
    with open(probe_info_path, 'r') as f:
        probe_info = json.load(f)
    probe_serial_number_found = False
    
    #TODO add more stuff: probe phase, ref channel etc
    if 'probe' in probe_info:
        probe_serial_number_found = True#probe_info['probe']
    else:
        failures.append((probe, 'Failed to find the probe serial number in :'+probe_info_path))


def check_timestamps_not_corrupted(probe, probe_files, failures, size_thresh=40000):
    timestamps_path = probe_files['event_timestamps.npy']
    timestamps_size = os.path.getsize(timestamps_path)
    if timestamps_size>size_thresh:#default is 40kb. might help check for truncated sorts too
        pass
    else:
        failures.append((probe, 'Event timestamps file too small'))


def check_npy_data_lengths(probe, probe_files, failures):
    
    metrics_path = probe_files['metrics.csv']
    metrics = pd.read_csv(metrics_path)
    metrics_units = len(metrics)

    spike_clusters = np.load(probe_files['spike_clusters.npy'])
    spike_clusters_units = len(np.unique(spike_clusters))
    if metrics_units != spike_clusters_units:
        failures.append((probe, 'Mismatch between number of units in metrics file, ' \
            'and spike clusters file'))
    
    cluster_length = len(spike_clusters)
    amplitudes = np.load(probe_files['amplitudes.npy'])
    spike_times = np.load(probe_files['spike_times.npy'])
    spike_templates = np.load(probe_files['spike_templates.npy'])

    if any([m.size!=cluster_length for m in (amplitudes, spike_times, spike_templates)]):
        failures.append((probe, 'Amplitudes, spike times, spike templates, ' \
            'and spike clusters must all be the same length'))
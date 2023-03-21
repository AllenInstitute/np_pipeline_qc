# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 16:01:29 2020

@author: svc_ccg
"""

D1_schema = {
    'area_classifications': {
        'rel_path': '*.areaClassifications.csv',
        'minimum_size': 611.2,
    },
    'beh_cam_json': {'rel_path': '*.behavior.json', 'minimum_size': 696.0},
    'behavior_stimulus': {
        'rel_path': '*.behavior.pkl',
        'minimum_size': 18426323.2,
    },
    'behavior_tracking': {
        'rel_path': '*.behavior.mp4',
        'minimum_size': 1956597323.2,
    },
    'EcephysBrainSurfaceLeft': {
        'rel_path': '*_surface-image2-left.png',
        'minimum_size': 1311537.6,
    },
    'EcephysBrainSurfaceRight': {
        'rel_path': '*_surface-image2-right.png',
        'minimum_size': 1316493.6,
    },
    'ephys_raw_data_probe_A': {
        'rel_path': '*probeABC\\*recording*npx2',
        'minimum_size': 407750128435.2,
    },
    'ephys_raw_data_probe_B': {
        'rel_path': '*probeABC\\*recording*npx2',
        'minimum_size': 407750128435.2,
    },
    'ephys_raw_data_probe_C': {
        'rel_path': '*probeABC\\*recording*npx2',
        'minimum_size': 407750128435.2,
    },
    'ephys_raw_data_probe_D': {
        'rel_path': '*probeDEF\\*recording*npx2',
        'minimum_size': 407750128435.2,
    },
    'ephys_raw_data_probe_E': {
        'rel_path': '*probeDEF\\*recording*npx2',
        'minimum_size': 407750128435.2,
    },
    'ephys_raw_data_probe_F': {
        'rel_path': '*probeDEF\\*recording*npx2',
        'minimum_size': 407750128435.2,
    },
    'eye_cam_json': {'rel_path': '*.eye.json', 'minimum_size': 689.6},
    'eye_tracking': {'rel_path': '*.eye.mp4', 'minimum_size': 1956543098.4},
    'face_cam_json': {
        'rel_path': '*.face.json',
        'minimum_size': 690.4000000000001,
    },
    'face_tracking': {'rel_path': '*.face.mp4', 'minimum_size': 1955874822.4},
    'fiducial_image': {
        'rel_path': '*.fiducial.png',
        'minimum_size': 1606833.6,
    },
    'EcephysInsertionLocationImage': {
        'rel_path': '*.insertionLocation.png',
        'minimum_size': 983416.0,
    },
    'isi _registration_coordinates': {
        'rel_path': '*.ISIregistration.npz',
        'minimum_size': 2926.4,
    },
    'newstep_csv': {'rel_path': '*.motor-locs.csv', 'minimum_size': 100.0},
    'optogenetic_stimulus': {
        'rel_path': '*.opto.pkl',
        'minimum_size': 959931.2000000001,
    },
    'EcephysOverlayImage': {
        'rel_path': '*.overlay.png',
        'minimum_size': 542164.0,
    },
    # 'post_experiment_surface_image_left': {'rel_path': '*_surface-image6-left.png',
    #  'minimum_size': 1308762.4000000001},
    # 'post_experiment_surface_image_right': {'rel_path': '*_surface-image6-right.png',
    #  'minimum_size': 1321449.6},
    # 'post_insertion_surface_image_left': {'rel_path': '*_surface-image4-left.png',
    #  'minimum_size': 1376120.0},
    # 'post_insertion_surface_image_right': {'rel_path': '*_surface-image4-right.png',
    #  'minimum_size': 1391095.2000000002},
    # 'post_stimulus_surface_image_left': {'rel_path': '*_surface-image5-left.png',
    #  'minimum_size': 1382569.6},
    # 'post_stimulus_surface_image_right': {'rel_path': '*_surface-image5-right.png',
    #  'minimum_size': 1393450.4000000001},
    # 'pre_experiment_surface_image_left': {'rel_path': '*_surface-image1-left.png',
    #  'minimum_size': 1400624.0},
    # 'pre_experiment_surface_image_right': {'rel_path': '*_surface-image1-right.png',
    #  'minimum_size': 1421704.0},
    # 'pre_insertion_surface_image_left': {'rel_path': '*_surface-image3-left.png',
    #  'minimum_size': 1334457.6},
    # 'pre_insertion_surface_image_right': {'rel_path': '*_surface-image3-right.png',
    #  'minimum_size': 1339092.8},
    'EcephysPostExperimentLeft': {
        'rel_path': '*_surface-image6-left.png',
        'minimum_size': 1308762.4000000001,
    },
    'EcephysPostExperimentRight': {
        'rel_path': '*_surface-image6-right.png',
        'minimum_size': 1321449.6,
    },
    'EcephysPostInsertionLeft': {
        'rel_path': '*_surface-image4-left.png',
        'minimum_size': 1376120.0,
    },
    'EcephysPostInsertionRight': {
        'rel_path': '*_surface-image4-right.png',
        'minimum_size': 1391095.2000000002,
    },
    'EcephysPostStimulusLeft': {
        'rel_path': '*_surface-image5-left.png',
        'minimum_size': 1382569.6,
    },
    'EcephysPostStimulusRight': {
        'rel_path': '*_surface-image5-right.png',
        'minimum_size': 1393450.4000000001,
    },
    'EcephysPreExperimentLeft': {
        'rel_path': '*_surface-image1-left.png',
        'minimum_size': 1400624.0,
    },
    'EcephysPreExperimentRight': {
        'rel_path': '*_surface-image1-right.png',
        'minimum_size': 1421704.0,
    },
    'EcephysPreInsertionLeft': {
        'rel_path': '*_surface-image3-left.png',
        'minimum_size': 1334457.6,
    },
    'EcephysPreInsertionRight': {
        'rel_path': '*_surface-image3-right.png',
        'minimum_size': 1339092.8,
    },
    'replay_stimulus': {
        'rel_path': '*.replay.pkl',
        'minimum_size': 27547471.200000003,
    },
    'surgery_notes': {
        'rel_path': '*_surgeryNotes.json',
        'minimum_size': 1314.4,
    },
    'synchronization_data': {'rel_path': '*.sync', 'minimum_size': 74143171.2},
    'visual_stimulus': {
        'rel_path': '*.mapping.pkl',
        'minimum_size': 7339021.600000001,
    },
}


# import os, sys, json, glob
#
# def get_file_size(file):
#
#    if file is None:
#        return
#
#    elif not os.path.exists(file):
#        print('File {} does not exist'.format(file))
#        return -1
#
#    file_size = os.path.getsize(file)
#    return file_size
#
#
# basedir = r"\\10.128.50.43\sd6.3\1041287144_522944_20200806"
# with open(r"\\10.128.50.43\sd6.3\1041287144_522944_20200806\1041287144_522944_20200806_platformD1.json", 'r') as f:
#    d1_platform = json.load(f)
#
#
# session_name = '1041287144_522944_20200806'
# d1_platform_files = d1_platform['files']
# paths = d1_platform_files
# schema = {k: {'rel_path':'', 'minimum_size':None} for k in paths}
# for p in paths:
#    file = paths[p]
#
#    if 'filename' in file:
#        filename = file['filename']
#        filesize = get_file_size(os.path.join(basedir, filename))
#        schema[p]['minimum_size'] = 0.8*filesize
#        schema[p]['rel_path'] = '*' + filename.split(session_name)[1]
#
#    elif 'directory_name' in file:
##        raw_data_dir = os.path.join(basedir, file['directory_name'])
##        raw_data_file = glob.glob(os.path.join(raw_data_dir, 'recording*npx2'))[0]
#        #filesize = get_file_size(raw_data_file)
#        filesize = get_file_size(r"\\10.128.50.43\sd6.3\1028043324_498757_20200604\1028043324_498757_20200604_probeDEF\recording_slot3_5.npx2")
#        schema[p]['minimum_size'] = 0.8*filesize
#        schema[p]['rel_path'] = '*probeABC\\*recording*npx2' \
#            if any([pid in p for pid in ['probe_A', 'probe_B', 'probe_C']]) else '*probeDEF\\*recording*npx2'

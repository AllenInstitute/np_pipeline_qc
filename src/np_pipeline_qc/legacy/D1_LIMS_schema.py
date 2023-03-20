# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:01:50 2020

@author: svc_ccg
"""

D1_schema = {
 'es_id': {'minimum_size': None},
 'es': {'minimum_size': None},
 'storage_directory': {'minimum_size': None},
 'workflow_state': {'minimum_size': None},
 'date_of_acquisition': {'minimum_size': None},
 'stimulus_name': {'minimum_size': None},
 'foraging_id': {'minimum_size': None},
 'external_specimen_name': {'minimum_size': None},
 'isi_experiment_id': {'minimum_size': None},
 'rig': {'minimum_size': None},
 'operator': {'minimum_size': None},
 'project': {'minimum_size': None},
 'behavior_dir': {'minimum_size': None},
 'EyeDlcOutputFile': {'minimum_size': 409683458.40000004},
 'EcephysPlatformFile': {'minimum_size': 5947.200000000001},
 'EcephysRigSync': {'minimum_size': 74143171.2},
 'EcephysSurgeryNotes': {'minimum_size': 1314.4},
 'EcephysReplayStimulus': {'minimum_size': 27547471.200000003},
 'OptoPickle': {'minimum_size': 959931.2000000001},
 'NewstepConfiguration': {'minimum_size': 100},
 'RawFaceTrackingVideo': {'minimum_size': 1955874822.4},
 'RawFaceTrackingVideoMetadata': {'minimum_size': 690.4000000000001},
 'RawEyeTrackingVideo': {'minimum_size': 1956543098.4},
 'RawEyeTrackingVideoMetadata': {'minimum_size': 689.6},
 'RawBehaviorTrackingVideo': {'minimum_size': 1956597323.2},
 'RawBehaviorTrackingVideoMetadata': {'minimum_size': 696.0},
 'EcephysAreaClassifications': {'minimum_size': 611.2},
 'mapping_pkl': {'minimum_size': 7339021.600000001},
 'replay_pkl': {'minimum_size': 27547471.200000003},
 'behavior_pkl': {'minimum_size': 18428066.400000002},
 'datestring': {'minimum_size': None},
 'data_probes': {'minimum_size': None},
 'EcephysProbeRawDataABC_settings': {'minimum_size': 1843772.0},
 'EcephysProbeRawDataABC': {'minimum_size': 378056894054.4},
 'EcephysProbeRawDataDEF_settings': {'minimum_size': 1843772.0},
 'EcephysProbeRawDataDEF': {'minimum_size': 378056264908.80005},
 'EcephysBrainSurfaceLeft': {'minimum_size': 1311537.6},
 'EcephysBrainSurfaceRight': {'minimum_size': 1316493.6},
 'EcephysFiducialImage': {'minimum_size': 1406833.6},
 'EcephysInsertionLocationImage': {'minimum_size': 900000.0},
 'EcephysOverlayImage': {'minimum_size': 542164.0},
 'EcephysPostExperimentLeft': {'minimum_size': 1308762.4000000001},
 'EcephysPostExperimentRight': {'minimum_size': 1321449.6},
 'EcephysPostInsertionLeft': {'minimum_size': 1376120.0},
 'EcephysPostInsertionRight': {'minimum_size': 1391095.2000000002},
 'EcephysPostStimulusLeft': {'minimum_size': 1382569.6},
 'EcephysPostStimulusRight': {'minimum_size': 1393450.4000000001},
 'EcephysPreExperimentLeft': {'minimum_size': 1400624.0},
 'EcephysPreExperimentRight': {'minimum_size': 1421704.0},
 'EcephysPreInsertionLeft': {'minimum_size': 1334457.6},
 'EcephysPreInsertionRight': {'minimum_size': 1339092.8},
}




D1_translator = {
 
 'EcephysPlatformFile': '*platformD1.json',
 'EcephysRigSync': '*.sync',
 'EcephysSurgeryNotes': '*surgeryNotes.json',
 'EcephysReplayStimulus': '*replay.pkl',
 'OptoPickle': '*opto.pkl',
 'NewstepConfiguration': '*motor-locs.csv',
 'RawFaceTrackingVideo': '*face.mp4',
 'RawFaceTrackingVideoMetadata': '*face.json',
 'RawEyeTrackingVideo': '*eye.mp4',
 'RawEyeTrackingVideoMetadata': '*eye.json',
 'RawBehaviorTrackingVideo': '*behavior.mp4',
 'RawBehaviorTrackingVideoMetadata': '*behavior.json',
 'EcephysAreaClassifications': '*areaClassifications.csv',
 'mapping_pkl': '*mapping.pkl',
 'replay_pkl': '*replay.pkl',
 'behavior_pkl': '*behavior.pkl',
 'EcephysProbeRawDataABC': '*probeABC',
 'EcephysProbeRawDataDEF': '*probeDEF',
 'EcephysBrainSurfaceLeft': '*surface-image1-left.png',
 'EcephysBrainSurfaceRight': '*surface-image1-right.png',
 'EcephysFiducialImage': '*fiducial.png',
 'EcephysInsertionLocationImage': '*insertionLocation.png',
 'EcephysOverlayImage': '*overlay.png',
 'EcephysPostExperimentLeft': '*surface-image2-left.png',
 'EcephysPostExperimentRight': '*surface-image2-right.png',
 'EcephysPostInsertionLeft': '*surface-image3-left.png',
 'EcephysPostInsertionRight': '*surface-image3-right.png',
 'EcephysPostStimulusLeft': '*surface-image4-left.png',
 'EcephysPostStimulusRight': '*surface-image4-right.png',
 'EcephysPreExperimentLeft': '*surface-image5-left.png',
 'EcephysPreExperimentRight':'*surface-image5-right.png',
 'EcephysPreInsertionLeft': '*surface-image6-left.png',
 'EcephysPreInsertionRight': '*surface-image6-right.png',
}








#
#schema = {k: {'minimum_size':None} for k in paths}
#for p in paths:
#    file = paths[p]
#    if not isinstance(file, str):
#        pass
#    elif os.path.isfile(file):
#        size = get_file_size(file)
#        schema[p]['minimum_size'] = 0.8*size
        
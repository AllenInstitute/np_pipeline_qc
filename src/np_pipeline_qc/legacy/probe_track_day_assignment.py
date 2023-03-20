# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:27:48 2021

@author: svc_ccg
"""
import pandas as pd
import numpy as np
import os, json
import get_sessions as gs
from matplotlib import pyplot as plt
import glob
import cv2
import query_lims
import probe_alignment_data_io as data_io


annotation_df = pd.read_excel(r"C:\Users\svc_ccg\ccb_onedrive\OneDrive - Allen Institute\VBN_annotation.xlsx")
angles = {'A': [-14.21405475, -12.3231693,  -58.84145942],
         'B': [ -17.34136572,  -13.18965862, -118.32166694],
         'C': [ -16.93653005,  -12.27583101, -177.7143598 ],
         'D': [-19.30100945, -16.39715103, 121.32239255],
         'E': [-16.82130366, -13.54745601,  61.47706882],
         'F': [-14.73266944, -13.27092408,   1.81126965]}
    
    
def yaw(inputMat, a): # rotation around z axis (heading angle)
    yawMat = np.array([[np.cos(a), -np.sin(a), 0],
               [np.sin(a), np.cos(a), 0],
               [0, 0, 1]])
    return np.dot(inputMat,yawMat)


def file_found(d, filestr):
    return len(glob.glob(os.path.join(d, filestr)))>0

source_volume_config = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\source_list.json"
with open(source_volume_config, 'r') as f:
    sources = json.load(f)

mice_to_run = ['583430']#, '569154', '565581', '568158']

ISI_pixels_per_micron = 0.44
#failed_mice = []
new_failed = []
multiple_session_dirs = []
for irow, row in annotation_df.iterrows():
    try:
        plt.close('all')
        #if row['animalID'] not in failed_mice or not row['Production List']:
        if not row['Production List']:
            continue
        
        mouseID = str(row['animalID'])
        
        if not mouseID in mice_to_run:
            continue
        
        print('generating images for mouse {}'.format(mouseID))
        sessions_to_run = gs.get_sessions(sources, mouseID=mouseID)
        if len(sessions_to_run) < 2:
            print('Found {} sessions'.format(sessions_to_run))
            raise ValueError('ERROR: Found {} sessions for mouse {} instead of expected 2'.format(len(sessions_to_run), mouseID))
            
        get_date = lambda x: os.path.basename(x).split('_')[-1][:8]
        get_limsID = lambda x: os.path.basename(x).split('_')[0]
        session_dates = np.array([get_date(s) for s in sessions_to_run])
        session_order = np.argsort(session_dates)
        
        session_dates = session_dates[session_order]
        session_limsids = np.array([get_limsID(s) for s in sessions_to_run])[session_order]
        sessions_to_run = np.array(sessions_to_run)[session_order]
        
        # If there are more than 2 directories returned, figure out which ones are valid and take those
        new_sessions_to_run = []
        
        if len(sessions_to_run)>2:
            multiple_session_dirs.append(mouseID)
            unique_lims_ids = np.unique(session_limsids)
            if mouseID=='527294':
                unique_lims_ids = unique_lims_ids[:2]
            for ulid in unique_lims_ids:
                session_dirs_to_check = [s for s in sessions_to_run if ulid in s]
                files_present = [all([file_found(s, fstr) for fstr in ['*area*.csv', 
                                      '*fiducial.png', '*surface-image3-left.png', 
                                      '*motor-locs.csv', '*probe*sorted']]) for s in session_dirs_to_check]
                session_to_use = [session_dirs_to_check[ind] for ind, f in enumerate(files_present) if f]
                if len(session_to_use)==0:
                    raise ValueError('ERROR: Could not find valid session dir among {}'.format(session_dirs_to_check))
                
                new_sessions_to_run.append(session_to_use[0])
            sessions_to_run = new_sessions_to_run
        
        
        # GET ISI TARGET IMAGE FROM LIMS AND DISPLAY INSERTION ANNOTATIONS
        SPECIMEN_QRY = '''
            SELECT id FROM specimens WHERE name LIKE '%{}';
            '''
        
        specimen_id = query_lims.query_lims(SPECIMEN_QRY.format(mouseID))[0]['id']
        
        ISI_QRY = '''
                    SELECT *
                    FROM isi_experiments
                    WHERE specimen_id = {};
        
        '''
        lims_results = query_lims.query_lims(ISI_QRY.format(specimen_id))
        valid_lims_entry = [l for l in lims_results if l['target_map_image_id']]
        if len(valid_lims_entry)==0:
            raise ValueError('ERROR: Could not find valid ISI experiment in lims for {}'.format(mouseID))
            
        isi_storage_dir = valid_lims_entry[0]['storage_directory']
        isi_storage_dir = '\\' + os.path.normpath(isi_storage_dir)
        target_image_path = glob.glob(os.path.join(isi_storage_dir, '*target_map.tif'))[0]
        
        target_image = cv2.imread(target_image_path)
        
        insertion_points = {}
        for ind, s in enumerate(sessions_to_run):
            
            insertion_annotations_file = glob.glob(os.path.join(s, '*area*.csv'))
            if len(insertion_annotations_file)>0:
                insertion_annotations = pd.read_csv(insertion_annotations_file[0])
                insertion_points[ind] = [insertion_annotations['ISI Pixel Coordinate X'].values,
                                     insertion_annotations['ISI Pixel Coordinate Y'].values]
            else:
                #if we can't find an area.csv file, try the ISIregistration npz file
                isireg_file = glob.glob(os.path.join(s, '*ISIregistration.npz'))
                if len(isireg_file)>0:
                    insertion_annotations = np.load(isireg_file[0])
                    insertion_pts = insertion_annotations['probeInsertionPointsTransformed']
                    insertion_points[ind] = [np.array([x[0] for x in insertion_pts]), np.array([x[1] for x in insertion_pts])]
                else:
                    raise ValueError('ERROR: Could not find valid probelocator annotations for {}'.format(s))
        
        # DISPLAY INSERTION IMAGES 
        fig, axes = plt.subplots(1,2)
        fig.set_size_inches([18,12])
        for ind, s in enumerate(sessions_to_run):
            insertion_im = glob.glob(os.path.join(s, '*fiducial.png'))
            if len(insertion_im)==0:
                raise ValueError('Could not find insertionLocation image in day {} session {}'.format(ind, s))
            
            im = cv2.imread(insertion_im[0])
            axes[ind].imshow(im[:, int(im.shape[1]/2):, ::-1])
        
        isifig, isiax = plt.subplots()
        isifig.set_size_inches([16,16])
        
        isiax.imshow(target_image)
        isiax.plot(insertion_points[0][0], insertion_points[0][1], 'r+', ms=10)
        isiax.plot(insertion_points[1][0], insertion_points[1][1], 'b+', ms=10)
        #isiax.legend(['Day 1', 'Day 2'])
        
        
        # GET INSERTION MOTOR COORDS FROM MOTOR LOC FILES
        # look for motor locs that are closest in time to the insertion image
        insertion_coords = []
        for ind, s in enumerate(sessions_to_run):
            image_file = data_io.glob_file(s, '*surface-image3-left.png')[0]
            insertion_time_stamp = data_io.get_modified_timestamp_from_file(image_file)
            
            motor_locs_file = data_io.glob_file(s, '*motor-locs.csv')[0]
            motor_locs = data_io.read_motor_locs_into_dataframe(motor_locs_file)
            session_date = os.path.basename(motor_locs_file).split('_')[-1][:8]
            motor_locs = motor_locs.loc[session_date]
            
            if mouseID in ['569156', '568158']: #weird motor assignments for two of the ultras
                SN_probe_dict = {' SN34027': 'B',
                                 ' SN31056': 'A',
                                 ' SN32141': 'E',
                                 ' SN32146': 'C',
                                 ' SN32139': 'D',
                                 ' SN32145': 'F'}
            
            else:
                SN_probe_dict = data_io.map_newscale_SNs_to_probes(motor_locs)
            
            session_insertion_coords = data_io.find_tap_coordinates(motor_locs,
                                                            SN_probe_dict,
                                                            insertion_time_stamp)
        
            insertion_coords.append(session_insertion_coords)
        
        
        motor_displacement = {}
        reticle_displacement = {}
        for p in insertion_coords[0]:
            d1 = np.array(insertion_coords[0][p])
            d2 = np.array(insertion_coords[1][p])
            motor_displacement[p] = d2-d1
            
            p_angles = -np.array(angles[p])*np.pi/180 + np.pi #flip angles around y for some reason
            
            #reticle_displacement[p] = probe_to_reticle([0,0,0], p_R, d2-d1)
            reticle_displacement[p] = yaw(d2-d1, p_angles[2])*ISI_pixels_per_micron 
        
        for ip, p in enumerate('ABCDEF'):
            reticle_d1 = [insertion_points[0][0][ip], insertion_points[0][1][ip]]
            displacement = reticle_displacement[p][:2]
            #isiax.plot([reticle_d1[0], reticle_d1[0]+displacement[0]], [reticle_d1[1], reticle_d1[1] - displacement[1]])
            isiax.arrow(reticle_d1[0], reticle_d1[1], displacement[0], -displacement[1], color='w', width=3, head_width=12)
        
        isiax.legend(['Day 1', 'Day 2'])
        isifig.suptitle('manual D1/D2 annotations for {}; \
                        arrows are inferred based on motor logs'.format(mouseID))
        fig.suptitle('D1/D2 insertion images for {}'.format(mouseID))
        
        if not isinstance(row['OPT directory'], str):
            raise ValueError('Could not find OPT directory for mouse {}'.format(mouseID))        
            
        save_dir = row['OPT directory']
            
        fig.savefig(os.path.join(save_dir, 'insertionImages.png'))
        isifig.savefig(os.path.join(save_dir, 'estimatedInsertions_corrected.png'))
    except Exception as e:
        new_failed.append((row['animalID'], e))
        #failed_mice.append((row['animalID'], e))
        print('failed to generate images for mouse {} due to error {}'.format(row['animalID'], e))

#fig, ax = plt.subplots()
#for p in reticle_displacement:
#    ax.plot([0, reticle_displacement[p][0]], [0, reticle_displacement[p][1]])
#    ax.text(reticle_displacement[p][0]+2, reticle_displacement[p][1]+2, p)
#ax.legend(['A', 'B', 'C', 'D', 'E', 'F'])
#ax.set_aspect('equal')

#motor_locs_file = r"\\10.128.54.20\sd8.2\1108528422_571520_20210610\1108528422_571520_20210610.motor-locs.csv"
#
#
#
#serialToProbeDict = {' SN32148': 'A', ' SN32142': 'B', ' SN32144':'C', ' SN32149':'D', ' SN32135':'E', ' SN24273':'F'}
#serialToProbeDict = {' SN34027': 'A', ' SN31056': 'B', ' SN32141':'C', ' SN32146':'D', ' SN32139':'E', ' SN32145':'F'}
#
##Date and time of experiment
#dateOfInterest = os.path.basename(motor_locs_file).split('_')[-1][:8]
##dateOfInterest = '2020-06-30'
#startTime = '0:00'  #I've set it to 12 am, only necessary to change if you did multiple insertions that day
#                    #This script just finds the first insertion after the experiment time
#
#fulldf = pd.read_csv(motor_locs_file, header=None, names=['time', 'probeID', 'x', 'y', 'z', 'relx', 'rely', 'relz'])
#fulldf['time'] = pd.to_datetime(fulldf['time'])
#fulldf = fulldf.set_index('time')
#
##Find probe trajectories for this experiment
#pdf = fulldf.loc[dateOfInterest]
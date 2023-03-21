# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:39:32 2021

@author: svc_ccg
"""

import glob
import json
import logging
import os
import re
import time
import urllib.request as request

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)


class bold_text:
    BOLD = '\033[1m'
    END = '\033[0m'


def get_exp_day1_session_dir(mouseID, basedir, output=True):
    sessions = get_session_dirs(mouseID, basedir, restrictions='exp_only')
    return get_session_idx(sessions, idx=0, output=True)


def get_last_session_directory(mouseID, basedir, output=True):
    sessions = get_session_dirs(mouseID, basedir)
    return get_session_idx(sessions, idx=-1, output=True)


def get_last_hab_dir(mouseID, basedir, output=True):
    sessions = get_session_dirs(mouseID, basedir, restrictions='hab_only')
    return get_session_idx(sessions, idx=-1, output=True)


def get_session_idx(sessions, idx=0, output=True):
    session_dates = [s.split('_')[-1] for s in sessions]
    sorted_inds = np.argsort(session_dates)
    session = sessions[sorted_inds[idx]]

    if output:
        print('\nUsing session: ' + session)

    return session


def get_session_dirs(mouseID, basedir, restrictions=None):
    mouseID = str(mouseID)

    sessions = glob_file(basedir, '*' + mouseID + '*')
    sessions = [
        s
        for s in sessions
        if validate_session_dir(s, restrictions=restrictions)
    ]
    if len(sessions) == 0:
        print(
            '\n Could not find session directories for mouse {} in {]'.format(
                mouseID, basedir
            )
        )
    return sessions


def validate_session_dir(d, restrictions=None):

    base = os.path.basename(d)
    match = re.search('[0-9]{10}_[0-9]{6}_[0-9]{8}', base)

    validated = True
    if restrictions == 'exp_only':
        filename = '*surface-image3*'
        look_for = os.path.join(d, filename)
        if not (len(glob.glob(look_for))):
            validated = False
    if restrictions == 'hab_only':
        filename = '*surface-image4*'
        look_for = os.path.join(d, filename)
        if len(glob.glob(look_for)):
            validated = False

    return (match is not None) and (validated)


def glob_file(basedir, file_string):

    result = []
    if isinstance(basedir, list):
        for b in basedir:
            this_result = glob.glob(os.path.join(b, file_string))
            result.extend(this_result)
    else:
        result = glob.glob(os.path.join(basedir, file_string))

    if len(result) == 0:
        print(
            '\nCould not find file of format {} '
            'in base directory {}'.format(file_string, basedir)
        )
    else:
        return result


def read_motor_locs_into_dataframe(motor_locs_csv_path):

    motor_locs = pd.read_csv(
        motor_locs_csv_path,
        header=None,
        names=['time', 'serialNum', 'x', 'y', 'z', 'relx', 'rely', 'relz'],
    )
    motor_locs['time'] = pd.to_datetime(motor_locs['time'])
    motor_locs = motor_locs.set_index('time')

    return motor_locs


def get_modified_timestamp_from_file(file, date_format='%Y/%m/%d %H:%M:%S'):

    # t = os.path.getctime(file)
    t = os.path.getmtime(file)
    t = time.localtime(t)
    t = time.strftime(date_format, t)

    return t


def map_newscale_SNs_to_probes(motor_locs):

    serial_numbers = motor_locs['serialNum'].unique()

    # Known serial number to probe mappings for NP rigs. Update here if new motors are added.
    NP0_serialToProbeDict = {
        ' SN32148': 'A',
        ' SN32142': 'B',
        ' SN32144': 'C',
        ' SN32149': 'D',
        ' SN32135': 'E',
        ' SN24273': 'F',
    }
    NP1_serialToProbeDict = {
        ' SN34027': 'A',
        ' SN31056': 'B',
        ' SN32141': 'C',
        ' SN32146': 'D',
        ' SN32139': 'E',
        ' SN32145': 'F',
    }
    NP3_serialToProbeDict = {
        ' SN31212': 'A',
        ' SN34029': 'B',
        ' SN31058': 'C',
        ' SN24272': 'D',
        ' SN32152': 'E',
        ' SN36800': 'F',
    }

    known_serial_to_probe_mapping = {}
    [
        known_serial_to_probe_mapping.update(d)
        for d in [
            NP0_serialToProbeDict,
            NP1_serialToProbeDict,
            NP3_serialToProbeDict,
        ]
    ]

    # Grab the probe mapping for all known serial numbers and leave unknown serial numbers unmapped
    try:
        assert all(
            [s in known_serial_to_probe_mapping for s in serial_numbers]
        )
    except Exception as e:
        unknown = []
        for s in serial_numbers:
            if s not in known_serial_to_probe_mapping:
                unknown.append(s)
                known_serial_to_probe_mapping[s] = ''
        warning_string = (
            '\nWARNING: Unknown newscale serial numbers {} encountered, '
            'please update serial number dictionary in data_io.py file'.format(
                unknown
            )
        )
        print(warning_string)
    finally:
        serialToProbeDict = {
            s: known_serial_to_probe_mapping[s] for s in serial_numbers
        }
        serialToProbeDict = {
            k: v
            for k, v in sorted(
                serialToProbeDict.items(), key=lambda item: item[1]
            )
        }
    print(
        '\nUsing following mapping between serial numbers and probe IDs: {}'.format(
            serialToProbeDict
        )
    )

    return serialToProbeDict


def find_tap_coordinates(motor_locs, serialToProbeDict, tap_time):

    tap_time = pd.to_datetime(tap_time)

    pcoordsDict = {}
    for pSN in serialToProbeDict:
        pid = serialToProbeDict[pSN]
        probe_locs = motor_locs.loc[motor_locs.serialNum == pSN]
        probe_locs['relz'] = (
            6000 - probe_locs['relz']
        )   # correct for weird z logging

        probe_locs = probe_locs.loc[(probe_locs.index < tap_time)]
        closest_motor_log_index = np.argmin(
            np.abs(probe_locs.index - tap_time)
        )
        closest_motor_log = probe_locs.iloc[closest_motor_log_index]
        # print('motor time: ', closest_motor_log)

        pcoordsDict[pid] = closest_motor_log[
            ['relx', 'rely', 'relz']
        ].to_list()

    return {pid: pcoordsDict[pid] for pid in 'ABCDEF' if pid in pcoordsDict}


def get_mouse_rig(mouseID, habituation_directories, raise_error=False):
    mouseID = str(mouseID)

    last_hab_session = get_last_session_directory(
        mouseID, habituation_directories, output=False
    )

    platform_json = glob_file(last_hab_session, '*platformD1.json')

    with open(platform_json[0]) as f:
        data = json.load(f)

    suggested_rig = data['rig_id']
    return suggested_rig


def verify_mouse_rig(mouseID, habituation_directories, rig, raise_error=False):

    suggested_rig = get_mouse_rig(
        mouseID, habituation_directories, raise_error
    )

    mouseID = str(mouseID)
    rig_match = all([c in suggested_rig for c in rig.upper()])

    if not rig_match:
        err_str = (
            '\n WARNING: Rig {} does not match rig {} from last '
            'habituation session for mouse {}\n'.format(
                rig, suggested_rig, mouseID
            )
        )
        print(bold_text.BOLD + err_str + bold_text.END)
        if raise_error:
            raise (AssertionError(err_str))
    else:
        print(
            '\nRig id validated: Specified rig matches last habituation session rig\n'
        )


def save_coords(coords, name, save_directory, append_datetime=True):

    coords_to_list = {k: list(v) for k, v in coords.items()}
    if append_datetime:
        now = time.localtime(time.time())
        now_str = time.strftime('%Y%m%d%H%M%S', now)
        name = name.split('.json')[0] + '_' + now_str + '.json'

    save_file = os.path.join(save_directory, name)
    with open(save_file, 'w') as f:
        json.dump(coords_to_list, f, sort_keys=True, indent=4)

    print('\nSaving coordinates to {}'.format(save_file))


def validate_LIMS_points(lims_points, name):

    if lims_points is False or len(lims_points) == 0:

        print('Could not find point on LIMS {}'.format(name))

    elif len(lims_points) == 6:

        print('6 LIMS points found on {} as expected'.format(name))

    else:

        print(
            bold_text.BOLD + '\n WARNING: Found {} points on LIMS '
            '{}\n'.format(len(lims_points), name) + bold_text.END
        )
        assert len(lims_points) == 6


# retrieve pixel coords from CSV
def get_pixel_coords(session_dir):
    newscale_csv = glob_file(session_dir, '*areaClassifications.csv')[0]
    # print(newscale_csv)
    current_csv_df = pd.read_csv(newscale_csv, index_col=False)
    coords = current_csv_df.loc[
        :, ['ISI Pixel Coordinate X', 'ISI Pixel Coordinate Y']
    ]
    # y_coords = current_csv_df.loc[:, 'ISI Pixel Coordinate Y']
    # print(x_coords, y_coords)
    coords = numpify(coords)
    return coords

    # need to get them in an appropriate array, will have to see what format is best


def make_appropriate_basis_array(array):
    num_points = np.shape(array)[0]
    # we add ones to account for the shift of the origin
    X = np.concatenate((np.ones([num_points, 1]), array), 1)
    return X


def fit_pixel_reticle_params(mouse_num):
    reticle_coords = get_isi_coords(
        mouse_num, line_type='insertion_targets', space_str='reticle_space'
    )
    reticle_coords = numpify_list_of_dicts(reticle_coords)
    pixel_coords = get_isi_coords(
        mouse_num, line_type='insertion_targets', space_str='image_space'
    )
    pixel_coords = numpify_list_of_dicts(pixel_coords)
    print(reticle_coords)
    print(pixel_coords)
    X = make_appropriate_basis_array(pixel_coords)
    print(X)
    y = reticle_coords  # [:, 0]
    print('y', y)
    # slove the linear system of equations
    betaHat = np.linalg.solve(X.T.dot(X), X.T.dot(y))
    print(betaHat)
    # show that we can recover y from only X
    est_y = X.dot(betaHat)
    print('est y', est_y)

    return betaHat


def numpify(iterable):
    if not (type(iterable) == np.ndarray):
        iterable = np.array(iterable)
    return iterable


def numpify_list_of_dicts(list_of_dicts):
    value_list = [list(element.values()) for element in list_of_dicts]
    return np.array(value_list)


def get_isi_coords(
    mouse_num, line_type='insertion_targets', space_str='reticle_space'
):
    json_string = request.urlopen(
        'http://lims2/specimens/isi_experiment_details/' + mouse_num + '.json'
    ).read()
    info = json.loads(json_string)

    experiments = info[0]['isi_experiments']
    found = False
    for exp in experiments:
        # print(exp['targets'][line_type])
        if exp['targets'][line_type][space_str] is not None:
            if found:
                raise (
                    AssertionError(
                        f'There are multiple maps with line type {line_type} on them'
                    )
                )
            # print('############################################'+str(exp['targets'].keys()))
            coords = exp['targets'][line_type][space_str]
            found = True
    return coords


def pixel_to_reticle(pixel_pts, mouse_num):
    R = fit_pixel_reticle_params(mouse_num)
    numpify(pixel_pts)
    X = make_appropriate_basis_array(pixel_pts)
    print(X)
    reticle_pt = np.dot(X, R)
    return reticle_pt


def mouse_num_from_session_dir(session_dir):
    location, dirname = os.path.split(session_dir)
    return dirname.split('_')[1]


def get_located_reticle_coords(session_dir):
    located_pixel_insertion_coords = get_pixel_coords(session_dir)
    mouse_num = mouse_num_from_session_dir(session_dir)
    located_reticle_insertion_coords = pixel_to_reticle(
        located_pixel_insertion_coords, mouse_num
    )

    # located_reticle_insertion_coords = []
    # for idx, probe in enumerate('ABCDEF'):
    #   pixel_pt = located_pixel_insertion_coords[idx]
    #   located_reticle_insertion_coords.append(pixel_to_reticle(pixel_pt, mouse_num))
    # located_reticle_insertion_coords = np.array(located_reticle_insertion_coords)
    return located_reticle_insertion_coords


def hab_dir_from_date(mouseID, tap_date, basedir, output=True):
    sessions = get_session_dirs(mouseID, basedir, restrictions='hab_only')

    return get_session_date(sessions, tap_date, output)


def get_session_date(sessions, date, output=True):
    session_dates = [s.split('_')[-1] for s in sessions]
    session = sessions[session_dates.index(date)]

    if output:
        print('\nUsing session: ' + session)

    return session

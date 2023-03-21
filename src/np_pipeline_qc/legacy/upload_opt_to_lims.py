# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 18:26:10 2020

@author: svc_ccg
"""

import argparse
import datetime
import glob
import json
import os
import shutil
import subprocess

import requests
from psycopg2 import connect, extras


def query_lims(query_string):

    con = connect(
        dbname='lims2',
        user='limsreader',
        host='limsdb2',
        password='limsro',
        port=5432,
    )
    con.set_session(
        readonly=True,
        autocommit=True,
    )
    cursor = con.cursor(
        cursor_factory=extras.RealDictCursor,
    )

    cursor.execute(query_string)
    result = cursor.fetchall()

    return result


def get_opt_data_dir(mouse_id, save_base):

    d = glob.glob(os.path.join(save_base, mouse_id))
    if len(d) == 0:
        d = None
    else:
        d = d[0] if os.path.isdir(os.path.join(save_base, d[0])) else None

    if d is None:
        print(
            'Could not find OPT data directory for {} in {}'.format(
                mouse_id, save_base
            )
        )

    return d


def get_acquisition_date(opt_data_dir):

    fluor_dir = os.path.join(os.path.join(opt_data_dir, 'trans'), 'native')
    img_file = glob.glob(os.path.join(fluor_dir, 'imgRot_0000.tif'))
    if len(img_file) == 0:
        print('could not find image data in {}'.format(fluor_dir))
        return
    else:
        m_date = os.path.getmtime(img_file[0])
        m_datetime = datetime.datetime.fromtimestamp(m_date)
        m_datetime_str = m_datetime.strftime('%Y-%m-%d')
        return m_datetime_str


#### QRY LIMS FOR REQUISITE INFO ####
def query_lims_for_ids(mouse_id, project_name, user_name):
    specimen_qry = """
    SELECT id FROM specimens WHERE name LIKE '%{}';
    """
    project_qry = """
    SELECT id FROM projects WHERE code = '{}';
    """
    user_qry = """
    SELECT id FROM users WHERE login = '{}';
    """

    specimen_id = query_lims(specimen_qry.format(mouse_id))[0]['id']
    project_id = query_lims(project_qry.format(project_name))[0]['id']
    operator_id = query_lims(user_qry.format(user_name))[0]['id']

    return specimen_id, project_id, operator_id


#### REST POST to LIMS to create DB entry ####
def post_to_lims(mouse_id, specimen_id, project_id, operator_id, date):
    url = 'http://lims2/observatory/opt_experiment/create'
    payload = {
        'name': date + '_' + mouse_id + '_opt',
        'specimen_id': str(specimen_id),
        'project_id': str(project_id),
        'operator_id': str(operator_id),
    }

    headers = {}
    res = requests.post(url, json=payload, headers=headers)
    return res


### make trigger file ###
def make_trigger_file(platform_save_dir, opt_lims_id, lims_incoming_dir):
    trigger_file = os.path.join(platform_save_dir, str(opt_lims_id) + '.tr2')
    print('Making trigger file: {}'.format(trigger_file))
    with open(trigger_file, 'w+') as triggerfile:
        L1 = r'experiment_id: {}'.format(opt_lims_id) + '\n'
        L2 = (
            r"location: '{}'".format(
                lims_incoming_dir + '/' + str(opt_lims_id) + '_platform.json'
            )
            + '\n'
        )
        L3 = r'type: opt_experiment' + '\n'

        [triggerfile.writelines(L) for L in [L1, L2, L3]]

    return trigger_file


### make platform json file ###
def make_platform_file(
    platform_save_dir,
    mouse_id,
    opt_lims_id,
    lims_incoming_dir,
    user_name,
    rig,
    acquisition_date,
):
    platform_file = os.path.join(
        platform_save_dir, str(opt_lims_id) + '_platform.json'
    )
    print('Making platform file: {}'.format(platform_file))
    contents = {
        'directory': lims_incoming_dir + '/' + mouse_id + '_OPT',
        'user_id': user_name,
        'rig_id': rig,
        'resolution': 6.9,
        'date_of_acquisition': acquisition_date,
    }

    with open(platform_file, 'w') as f:
        json.dump(contents, f)

    return platform_file


### copy data to incoming folder ###
def transfer_data(mouse_id, opt_data_dir, lims_incoming_dir):
    lims_path = '\\' + os.path.normpath(lims_incoming_dir)
    dest_dir = os.path.join(lims_path, mouse_id + '_OPT')
    command_string = 'robocopy ' + opt_data_dir + ' ' + dest_dir + r' /E'
    print(command_string)
    P = subprocess.call(command_string)
    print(
        'Copied {} to {} with return code {}'.format(opt_data_dir, dest_dir, P)
    )


### copy trigger and platform files ###
def transfer_trigger_platform(
    trigger_dir, lims_incoming_dir, trigger_file, platform_file
):
    trigger_dir_norm = '\\' + os.path.normpath(trigger_dir)
    lims_incoming_norm = '\\' + os.path.normpath(lims_incoming_dir)
    shutil.copyfile(
        trigger_file,
        os.path.join(trigger_dir_norm, os.path.basename(trigger_file)),
    )
    shutil.copyfile(
        platform_file,
        os.path.join(lims_incoming_norm, os.path.basename(platform_file)),
    )


def main(mouse_id, opt_save_base, project_name, user_name, rig):

    #    mouse_id = '506940'
    #    opt_save_base = r"\\10.128.50.20\sd7\OPT"
    #    project_name = 'NeuropixelVisualBehaviorDevelopment'
    #    user_name = 'corbettb'
    #    rig = 'OPT.0'
    date = datetime.datetime.now().strftime('%Y%m%d')

    ### GIT LIMS IDs for specimen, project and operator to put in meta files
    specimen_id, project_id, operator_id = query_lims_for_ids(
        mouse_id, project_name, user_name
    )

    ## Find local data directory for this mouse and get imaging date
    opt_data_dir = get_opt_data_dir(mouse_id, opt_save_base)
    acquisition_date = get_acquisition_date(opt_data_dir)

    ### specify where to save the trigger and platform files for our reference (these will get copied to the lims incoming dir)
    platform_save_base = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\OPT_lims'
    platform_save_dir = os.path.join(platform_save_base, mouse_id)
    if not os.path.exists(platform_save_dir):
        os.mkdir(platform_save_dir)

    ### POST to lims to create entry for this data ###
    res = post_to_lims(mouse_id, specimen_id, project_id, operator_id, date)
    opt_lims_id = res.json()['id']
    trigger_dir = res.json()['trigger_dir']
    lims_incoming_dir = os.path.dirname(os.path.dirname(trigger_dir))

    trigger_file = make_trigger_file(
        platform_save_dir, opt_lims_id, lims_incoming_dir
    )
    platform_file = make_platform_file(
        platform_save_dir,
        mouse_id,
        opt_lims_id,
        lims_incoming_dir,
        user_name,
        rig,
        acquisition_date,
    )

    transfer_data(mouse_id, opt_data_dir, lims_incoming_dir)
    transfer_trigger_platform(
        trigger_dir, lims_incoming_dir, trigger_file, platform_file
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mouse_id', help='mouse ID (6 digit labtracks id)')

    parser.add_argument(
        '-s',
        '--source',
        help='source directory for OPT data (probably on NAS)',
        default=r'\\10.128.54.20\sd8.2\OPT',
    )

    parser.add_argument(
        '-p',
        '--project',
        help='LIMS project name associated with experiment',
        default='NeuropixelVisualBehaviorDevelopment',
    )

    parser.add_argument(
        '-u', '--user', help='Operator name (allen login)', default='corbettb'
    )

    parser.add_argument(
        '-r',
        '--rig',
        help='OPT rig on which data was collected (OPT.0 or OPT.1)',
        default='OPT.0',
    )

    args = parser.parse_args()
    mouse_id = args.mouse_id
    opt_save_base = args.source
    project_name = args.project
    user_name = args.user
    rig = args.rig

    main(mouse_id, opt_save_base, project_name, user_name, rig)

#    #mouse_id = '506940'
#    mouse_id = sys.argv[1]
#    opt_save_base = r"\\10.128.50.20\sd7\OPT"
#    project_name = 'NeuropixelVisualBehaviorDevelopment'
#    user_name = 'corbettb'
#    rig = 'OPT.0'
#

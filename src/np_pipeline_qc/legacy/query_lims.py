# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 12:30:43 2020

@author: svc_ccg
"""

from psycopg2 import connect, extras

DONOR_FROM_LABTRACKS_QRY = """
    SELECT *
    FROM donors d
    WHERE d.external_donor_name=cast({} as character varying)
    """

DONOR_QRY = """
    SELECT *
    FROM donors d
    WHERE d.id = {}
    """

BEHAVIOR_SESSION_QRY = """
    SELECT *
    FROM behavior_sessions bs
    WHERE bs.id={}
    """

PROJECT_QRY = """
    SELECT *
    FROM projects p
    WHERE p.id={}
    """

ECEPHYS_SESSION_QRY = """
    SELECT *
    FROM ecephys_sessions es
    WHERE es.id = {}
    """

OPT_QRY = """
    SELECT *
    FROM opt_experiments op
    WHERE op.specimen_id = {}
    """

TISSUECYTE_QRY = """
    SELECT *
    FROM image_series im
    WHERE im.specimen_id = {}
    """

SPECIMEN_QRY = """
    SELECT *
    FROM specimens sp
    WHERE sp.external_specimen_name=cast({} as character varying)
    """

# BEHAVIOR_SESSION_BY_DATEQRY = '''
#    SELECT *
#    FROM behavior_sessions bs
#    WHERE bs.date_of_acquisition > '9/01/2022 12:00:00 AM'
#    '''


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


def get_labtracks_id_from_donor_id(donor_id):

    donor_info = query_lims(DONOR_QRY.format(donor_id))
    return donor_info[0]['external_donor_name']


def get_donor_id_from_labtracks_id(labtracks_id):
    mouse_info = query_lims(DONOR_FROM_LABTRACKS_QRY.format(int(labtracks_id)))
    return mouse_info[0]['id']


def get_specimen_id_from_labtracks_id(labtracks_id):

    mouse_info = query_lims(SPECIMEN_QRY.format(int(labtracks_id)))
    return mouse_info[0]['id']


def get_project_from_labtracks_id(labtracks_id):

    mouse_info = query_lims(SPECIMEN_QRY.format(int(labtracks_id)))
    project_id = mouse_info[0]['project_id']
    project_info = query_lims(PROJECT_QRY.format(project_id))

    return project_info[0]['code']

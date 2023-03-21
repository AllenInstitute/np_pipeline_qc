# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 09:44:09 2020

@author: svc_ccg
"""
import json
import os
import sys

from psycopg2 import connect, extras

from np_pipeline_qc.legacy import data_getters
from np_pipeline_qc.legacy.D1_LIMS_schema import D1_schema
from np_pipeline_qc.legacy.D2_LIMS_schema import D2_schema


def run_validation(lims_id, savePath=None):
    '''Top level wrapper for lims validation. 
        INPUT:
            lims_id: lims id for Extracellular Ephys Session (EcephysSession)
            savePath: where to save validation report json. 
                If none, report isn't saved
        
        OUTPUT:
            master_report: dictionary (JSON serializable) with 3 keys:
                D1_upload_summary: Summary of whether the Day 1 LIMS upload succeeded. Contains 3 keys:
                        upload_exists: Boolean indicating whether any D1 data is in LIMS for this session
                        pass: Boolean indicating whether D1 data agreed with D1_schema. To pass
                                1) all files specified by schema must exist, and 
                                2) they must be larger than the 'min_size' specified in schema
                        errors: list of failures indicating which files didn't exist
                                and which files failed to meet size requirement
                D2_upload_summary: Same as D1_report but for Day 2 LIMS upload.
                
                file_validation: primary data object for each session. Contains details for 
                    each file from which the D1 and D2 reports are built. Output of lims_validation.
    '''
    #set default report to false
    D1_report = {'upload_exists':False}
    D2_report = {'upload_exists': False}
    
    #basic check of lims session info
    (exists_in_lims, has_storage_directory, \
     storage_dir_has_data, valid_foraging_id) = check_lims_for_session(lims_id)
    
    if not exists_in_lims:
        validator = {'Error':'No entry in lims'}
    
    elif not has_storage_directory:
        validator = {'Error': 'Storage directory not yet assigned'}
    
    elif not storage_dir_has_data:
        validator = {'Error': 'Storage directory empty. Data not yet uploaded'}
    
    elif not valid_foraging_id:
        validator = lims_validation(lims_id)
        validator['Error'] = 'No foraging ID assigned to experiment'
        D1_report = upload_summary(validator['D1'])
        D2_report = upload_summary(validator['D2'])
#        validator = {'Error': 'No foraging ID assigned to experiment'}
#        D1_report = {'upload_exists':True, 'errors':'No foraging ID assigned to experiment'}
#        D2_report = {'upload_exists':True, 'errors':'No foraging ID assigned to experiment'}
    
    #if it looks like the session exists in lims and has data then do some data checking    
    else:
        #this validator checks for expected files and verifies they are the appropriate size
        validator = lims_validation(lims_id)
        
        #TODO: make this error handling more helpful
        if validator['Error'] is not None:
            error_string = 'LIMS entry found but error getting data. Check file_validation for details'
            D1_report = {'upload_exists':False, 'errors':error_string}
            D2_report = {'upload_exists':False, 'errors':error_string}
        else:
            D1_report = upload_summary(validator['D1'])
            D2_report = upload_summary(validator['D2'])
    
    master_report = {
                'D1_upload_summary': D1_report,
                'D2_upload_summary': D2_report,
                'file_validation':validator,
                         }
    if savePath:
        save_json(master_report, savePath)
    
    return master_report

    
def lims_validation(lims_id):
    '''Validation function to check LIMS session for files associated with the Day1 and Day2 uploads.
        Uses D1 and D2 schema to define
            1) which files to looks for, and
            2) how big they should be (right now just a min_size threshold defined as 80% of the size
                 of a random good session. This could be improved...)
        If you would like to add files to this validation, edit the D1_LIMS_schema, or D2_LIMS_schema
        files (imported at top).
        
        INPUT: 
            lims_id: lims ecephys session id
        OUTPUT:
            lims_validator: dictionary with 4 keys:
                storage_directory: storage directory for session in LIMS
                D1: results of checking LIMS data against the D1_schema (output of check_schema)
                D2: results of checking LIMS data against the D2_schema
                Error: reporting of any errors encountered while attempting to run
                    validation on data
    '''
    try:
        d = data_getters.lims_data_getter(lims_id)
        paths = d.data_dict
        storage_dir = os.path.normpath(paths['storage_directory'])
        lims_validator = {'storage_directory': storage_dir,
                          'D1':{}, 'D2':{}}
        lims_validator['D1'] = check_schema(D1_schema, paths)
        lims_validator['D2'] = check_schema(D2_schema, paths)
        lims_validator['Error'] = None
    
    except:
        #TODO: This error handling isn't great. Difficult to traceback.
        lims_validator = {'Error':str(sys.exc_info()[0]) + 
                          '  ' + str(sys.exc_info()[1]) + 
                          '  ' + str(sys.exc_info()[2])}
    
    return lims_validator


def check_schema(schema, paths):
    '''Check LIMS session against pre-defined schema. 
        INPUT:
            schema: Dictionary specifying which files to look for and their
                minimum acceptable size (must exceed this number to pass)
            paths: paths in LIMS to files associated with a session
                (usually generated by the data_getters class imported at top)
        OUTPUT:
            validation_dict: dictionary specifying whether each file exists
                and meets the size criterion
    '''
    validation_dict = {}
    for key in schema:

        (meets_size_criterion, size, criterion) = validate_schema_entry_size(schema, key, paths)
        
        validation_dict[key] = {
                'exists': validate_schema_entry_existence(schema, key, paths),
                'file_size': size,
                'min_expected_size': criterion,
                'meets_size_criterion': meets_size_criterion}   
    
    return validation_dict


def upload_summary(validator):
    '''Summarize results of check schema
    '''
    report = {'pass': False, 'errors':[]}
    
    exists = []
    meets_size = []
    for entry in validator:
        
        ex = validator[entry]['exists']
        if not ex:
            report['errors'].append('File {} does not exist'.format(entry))
        exists.append(ex)
        
        ms = validator[entry]['meets_size_criterion']
        if ex and not ms:
            report['errors'].append('File {} does not meet size criterion'.format(entry))
        meets_size.append(ms)
    
    report['pass'] = all(exists) & all(meets_size)
    report['upload_exists'] = any(exists)
    
    return report
    

def validate_schema_entry_existence(schema, entry, paths):
    '''Check that particular file in schema exists in LIMS session data
    '''
    if entry not in paths:
        return False
    elif paths[entry] is None:
        return False
    elif schema[entry]['minimum_size'] is not None and isinstance(paths[entry], str):
        return os.path.exists(paths[entry])
    else:
        return True
                    

def validate_schema_entry_size(schema, entry, paths):
    '''Check that particular file in LIMS session data meets size requirement
        specified by schema
    '''
    min_size = schema[entry]['minimum_size']
    if not min_size:
        return (True, None, None)
    elif not entry in paths or paths[entry] is None:
        return(False, None, None)
    else:
        file_size = get_file_size(paths[entry])
        return (file_size > min_size, file_size, min_size)
    

def check_lims_for_session(lims_id):
    '''Basic check that LIMS session data
        1) exists
        2) has a storage directory assigned
        3) storage directory actually has data in it
        4) session has an associated foraging ID
    '''
    ECEPHYS_QRY = '''
    SELECT *
    FROM ecephys_sessions es
    WHERE es.id = {} 
    ORDER BY es.id
    '''
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

    cursor.execute(ECEPHYS_QRY.format(lims_id))
    lims_session = cursor.fetchall()
    
    session_exists = len(lims_session)>0
    storage_directory_exists = False
    storage_directory_has_files = False
    assigned_foraging_id = False
    if session_exists:
        storage_directory_exists = lims_session[0]['storage_directory'] is not None
        assigned_foraging_id = lims_session[0]['foraging_id'] is not None
    if storage_directory_exists:
        storage_directory = '\\'+os.path.normpath(lims_session[0]['storage_directory'])
        storage_directory_has_files = len(os.listdir(storage_directory))>0
        
    return session_exists, storage_directory_exists, storage_directory_has_files, assigned_foraging_id
    

def get_file_size(file):
    
    if file is None:
        return
    
    elif not os.path.exists(file):
        print('File {} does not exist'.format(file))
        return -1
    
    file_size = os.path.getsize(file)
    return file_size


def save_json(to_save, save_path):
    
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    with open(save_path, 'w') as f:
        json.dump(to_save, f, indent=2)     
        
  
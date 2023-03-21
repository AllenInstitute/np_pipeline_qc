# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 16:55:19 2020

@author: svc_ccg
"""

import glob
import json

# from D2_LIMS_schema import D2_schema
import os
import sys

from np_pipeline_qc.legacy.D1_local_schema import D1_schema


def run_local_validation(local_dir):
    schema_names = ['D1']
    schemas = [D1_schema]

    lims_upload_report = {s: None for s in schema_names}
    for schema_name, schema in zip(schema_names, schemas):

        report = upload_ready_validation(local_dir, schema)
        lims_upload_report[schema_name] = report

    return report


def upload_ready_validation(local_dir, schema):

    report = {'pass': False, 'Errors': [], 'files': {e: {} for e in schema}}
    for name, file in schema.items():
        exists = validate_schema_entry_existence(local_dir, file)
        report['files'][name]['exists'] = exists
        (meets_criterion, file_size, min_size) = validate_schema_entry_size(
            local_dir, file
        )
        report['files'][name]['meets_size_criterion'] = meets_criterion
        report['files'][name]['file_size'] = file_size
        report['files'][name]['min_expected_size'] = min_size

        if not exists:
            report['Errors'].append('File {} does not exist'.format(name))
        elif not meets_criterion:
            report['Errors'].append(
                'File {} does not meet size criterion'.format(name)
            )

    if len(report['Errors']) == 0:
        report['pass'] = True

    return report


def validate_schema_entry_existence(local_dir, schema_file):
    """Check that particular file in schema exists in local dir"""

    rel_path = schema_file['rel_path']
    glob_result = glob_file(local_dir, rel_path)

    if glob_result is None:
        return False
    else:
        return True


def validate_schema_entry_size(local_dir, schema_file):
    """Check that particular file in local dir meets size requirement
    specified by schema
    """
    min_size = schema_file['minimum_size']
    if not min_size:
        return (True, None, None)
    elif not validate_schema_entry_existence(local_dir, schema_file):
        return (False, None, None)
    else:
        file_size = get_file_size(
            glob_file(local_dir, schema_file['rel_path'])
        )
        return (file_size > min_size, file_size, min_size)


def get_file_size(file):

    if file is None:
        return

    elif not os.path.exists(file):
        print('File {} does not exist'.format(file))
        return -1

    file_size = os.path.getsize(file)
    return file_size


def glob_file(root, format_str):

    f = glob.glob(os.path.join(root, format_str))
    if len(f) > 0:
        return f[0]
    else:
        print(
            'Could not find file of format' '{} in {}'.format(format_str, root)
        )
        return None

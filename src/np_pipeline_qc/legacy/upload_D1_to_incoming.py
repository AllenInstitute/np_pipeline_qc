# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:12:27 2022

@author: svc_ccg
"""

from D1_LIMS_schema import D1_translator
import glob, os, subprocess, json

sessions_to_run = []

s = r'\\10.128.50.43\sd6.3\1043752325_506940_20200817'
lims_incoming = r'\\allen\programs\braintv\production\incoming\neuralcoding'

manifest = []
for file in D1_translator:
    
    local_file = glob.glob(os.path.join(s, D1_translator[file]))
    try:
        manifest.append(local_file[0])
    except:
        print('could not find {} in {}'.format(file, s))
        

#cross check with D1 platform json
platform_json_file = glob.glob(os.path.join(s, '*platformD1.json'))[0]
with open(platform_json_file, 'r') as f:
    platform_info = json.load(f)

def unpack_dict(d):
    contents = []
    for key in d:
        #print(key)
        if isinstance(d[key], dict):
            c = unpack_dict(d[key])
        else:
            c = d[key]
        contents.append(c)
        
    return contents
    

platform_manifest = unpack_dict(platform_info['files'])
platform_manifest = [l[0] for l in platform_manifest]
manifest_base_filenames = [os.path.basename(f) for f in manifest]
for platform_file in platform_manifest:
    if platform_file not in manifest_base_filenames:
        if len(glob.glob(os.path.join(s, platform_file)))>0:
            print('Adding {} to manifest'.format(platform_file))
            manifest.append(os.path.join(s, platform_file))
        else:
            print('File {} in platform json but not manifest. Cannot find it in base dir {}'.format(platform_file, s))
    
    



for file in manifest:
    
    print('copying {} to {}'.format(file, lims_incoming))
    source_dir = os.path.dirname(file)
    dest_dir = lims_incoming
    filename = os.path.basename(file)
    
    if os.path.isdir(file):
        command_string = "robocopy "+ file +" " + os.path.join(dest_dir, filename) + r" /xc /xn /xo /e"
        print(command_string)
    else:
        command_string = "robocopy "+ source_dir +" " + dest_dir + " " + filename + r" /xc /xn /xo"
    
    
    P = subprocess.call(command_string)
    
    
    
    

ECHO off
title Script to upload raw OPT data to LIMS
ECHO activating environment: base
call C:\Users\svc_ccg\AppData\Local\Continuum\anaconda3\Scripts\activate.bat C:\Users\svc_ccg\AppData\Local\Continuum\anaconda3\
call conda activate base
ECHO navigating to code directory
call cd C:\Users\svc_ccg\Documents\GitHub\NP_pipeline_QC
call python upload_opt_to_lims.py -h
set /p command_string=Command string, eg mouse_id [-s source_dir][-u user][-p project][-r rig]: 
call python -W ignore upload_opt_to_lims.py %command_string%
cmd \k
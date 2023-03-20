ECHO off
title Script to get RFs given data directory
ECHO activating environment: np_pipeline_validation_clone
call conda activate np_pipeline_validation_clone
ECHO navigating to code directory
call cd C:\Users\svc_neuropix\Documents\GitHub\NP_pipeline_QC
set /p data_dir=Base data directory: 
call python get_RFs_standalone.py %data_dir%
cmd \k

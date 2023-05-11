ECHO off
title Script to run post-experiment QC
cd C:\Users\svc_neuropix\Documents\GitHub\np_pipeline_qc2
ECHO activating environment
call .venv\scripts\activate.bat
ECHO All QC modules will be run
setlocal EnableDelayedExpansion
set session=%1
echo received input %session%
if "%~1" == "" set /p session=Enter session lims id or foldername or path: 
call python -m np_pipeline_qc %session%
if "%~1" == "" cmd /k

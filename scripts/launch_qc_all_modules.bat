ECHO off
title Script to run post-experiment QC
ECHO activating environment
call conda activate np_pipeline_qc
ECHO All QC modules will be run
setlocal EnableDelayedExpansion
set session=%1
echo received input %session%
if "%~1" == "" set /p session=Enter session lims id or foldername or path: 
call python -m np_pipeline_qc %session%
if "%~1" == "" cmd /k

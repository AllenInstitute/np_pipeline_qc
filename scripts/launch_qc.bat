ECHO off
title Script to run post-experiment QC
ECHO activating environment: pipelineQC
call C:\Users\svc_ccg\AppData\Local\Continuum\anaconda3\Scripts\activate.bat C:\Users\svc_ccg\AppData\Local\Continuum\anaconda3\
call conda activate base
ECHO navigating to code directory
call cd C:\Users\svc_ccg\Documents\GitHub\NP_pipeline_QC
call python qc_cmd_caller.py -h
set /p command_string=Command string, eg session path [-p PROBES] [-ctx]: 
call python -W ignore qc_cmd_caller.py %command_string%
cmd \k

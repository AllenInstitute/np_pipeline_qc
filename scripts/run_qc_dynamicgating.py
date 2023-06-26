import shutil
import pathlib

import np_tools
import np_session

from np_pipeline_qc.legacy import run_qc_class

for s in np_session.Projects.DRDG.state['ephys']:
    session = np_session.Session(s)
    
    print(session.qc_path)
    
    path: pathlib.Path
    for path in session.qc_paths[0].rglob('*'):
        if path.is_dir():
            continue
        copy: pathlib.Path = session.qc_path / path.relative_to(session.qc_paths[0])
        if copy.exists() and (copy.is_dir() and not path.is_dir()):
            shutil.rmtree(copy)
        if not copy.exists():
            copy.mkdir(exist_ok=True, parents=True)
            shutil.copy(path, copy.parent)
    # try:
    #     if not (session.qc_path / 'probe_yield' / 'probe_depth').exists():
    #         run_qc_class.DR1(str(session.npexp_path), str(session.qc_path), modules_to_run='spike_depths')
    # except PermissionError:
    #     pass 
    
    # if not (session.qc_path / 'probe_yield' / 'probe_depth').exists():
    #     print(f'Spike depths still don\'t exist: {session.folder}')
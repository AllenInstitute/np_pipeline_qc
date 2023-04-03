import time
from matplotlib import pyplot as plt

import np_logging
import np_session

import np_pipeline_qc


# plt.style.use('seaborn-v0_8-whitegrid')

if __name__ == '__main__':
    logger = np_logging.getLogger()
    sessions = sorted(np_session.sessions(project='TTN'))
    np_pipeline_qc.run_qc(sessions[-1], debug=False)
    exit()
    
    for session in np_session.sessions(project='TTN'):
        if session.mouse == 366122:
            continue
        np_pipeline_qc.run_qc(session, modules_to_run=('all',), debug=True)

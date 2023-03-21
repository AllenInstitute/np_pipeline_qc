import argparse

import np_logging
import np_session

from np_pipeline_qc.classes import BaseQC

if __name__ == '__main__':

    logger = np_logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'session',
        type=np_session.Session,
        help='A lims session ID, or a string/path containing one',
    )
    session = parser.parse_args(
        tuple(str(np_session.Projects.TTN.get_latest_ephys()))
    ).session

    logger.info(
        f'Running QC for {session} | {"Hab" if session.is_hab else "Ephys"} | {session.project}'
    )

    # make QC-class factory
    BaseQC(session).run()

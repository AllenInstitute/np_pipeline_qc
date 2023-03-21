import argparse
from typing import Type

import np_logging
import np_session

from np_pipeline_qc.classes import BaseQC
from np_pipeline_qc.legacy import run_qc_class

logger = np_logging.getLogger(__name__)


def run_qc(session: int | str | np_session.Session) -> None:
    if not isinstance(session,  np_session.Session):
        session = np_session.Session(session)

    logger.info(
        f'Running QC for {session} | {"Hab" if session.is_hab else "Ephys"} | {session.project}'
    )

    cls: Type[run_qc_class.run_qc]
    
    if session.is_hab:
        cls = run_qc_class.run_qc_hab
    elif session.project in np_session.Projects.DR.value:
        cls = run_qc_class.DR1
    else:
        cls = run_qc_class.run_qc_passive
    
    # instantiating runs qc
    # str(`id`) > lims 
    cls(str(session.npexp_path), str(session.qc_path.parent))
    
    
if __name__ == '__main__':

    logger = np_logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'session',
        type=np_session.Session,
        help='A lims session ID, or a string/path containing one',
    )
     
    #! session = parser.parse_args().session
    session = np_session.Session(np_session.Projects.TTN.get_latest_ephys())

    run_qc(session)

    # TODO finish QC-class factory:
    # def qc_factory(session: int | str | np_session.Session) -> BaseQC:
    #     if not isinstance(session,  np_session.Session):
    #         session = np_session.Session(session)
    #     match session:
    #         case session.is_hab:
    #             return HabQC(session)
    #         case session.project :
    #     return BaseQC(session)
    # BaseQC(session).run()

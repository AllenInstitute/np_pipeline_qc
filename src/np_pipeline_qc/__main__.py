import argparse

import np_session

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('session', type=np_session.Session, help='A lims session ID, or a string/path containing one')
    session = parser.parse_args().session

    print(f'Running QC for {session} | {"Hab" if session.is_hab else "Ephys"} | {session.project}')
    print('[not implemented]')

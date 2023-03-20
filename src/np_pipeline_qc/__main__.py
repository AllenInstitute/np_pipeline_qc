import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('session', type=pathlib.Path, required=True, help='A lims session ID, or a string/path containing one')
args = parser.parse_args()

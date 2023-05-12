from __future__ import annotations

import argparse
import concurrent.futures
import doctest
import functools
import pathlib
import re
from typing import Generator, Literal, Optional

import matplotlib.pyplot as plt
import numba
import numpy as np
import numpy.typing as npt
import np_logging

logger = np_logging.getLogger(__name__)

GLOB_AP_DIR = '**/continuous/Neuropix-PXI-???.0'
"""Glob pattern for finding AP directories in a parent directory containing Kilosort
output subfolders."""


def sorted_continuous_AP_dirs(
    parent: str | pathlib.Path,
) -> Generator[pathlib.Path, None, None]:
    """Generator yielding any sorted AP directories within any parent directory.

    >>> next(sorted_continuous_AP_dirs('//allen/programs/mindscope/production/variability/prod0/specimen_1153809254/ecephys_session_1176583610/1184386053/1176693482_probeA/continuous/Neuropix-PXI-100.0')).as_posix()
    '//allen/programs/mindscope/production/variability/prod0/specimen_1153809254/ecephys_session_1176583610/1184386053/1176693482_probeA/continuous/Neuropix-PXI-100.0'
    >>> next(sorted_continuous_AP_dirs('//allen/programs/mindscope/production/variability/prod0/specimen_1153809254/ecephys_session_1176583610/1184386053')).as_posix()
    '//allen/programs/mindscope/production/variability/prod0/specimen_1153809254/ecephys_session_1176583610/1184386053/1176693482_probeA/continuous/Neuropix-PXI-100.0'
    """
    parent = pathlib.Path(parent)
    if parent.match(GLOB_AP_DIR):
        yield parent
    yield from parent.glob(GLOB_AP_DIR)
    

def get_all_probe_letter_groups_from_path(probe_string: str | pathlib.Path) -> tuple[str, ...]:
    """Extract all groups of probe letters from a path-like containing `_probe`.
    
    - no validation that the groups make sense. For example, `probeA_probeB` will be
    returned as `('A', 'B')`.
    
    - if no probe letters are found, an empty tuple is returned.
    
    >>> get_all_probe_letter_groups_from_path('1184386053/1176693482')
    ()
    >>> get_all_probe_letter_groups_from_path('1184386053/1176693482_probeA')
    ('A',)
    >>> get_all_probe_letter_groups_from_path(pathlib.Path('1184386053/1176693482_probeABC/1176693482_probeA'))
    ('ABC', 'A')
    """
    probe_string = str(probe_string)
    matches = re.findall(r'(?<=_probe)[A-F]+', probe_string)
    return tuple(matches)

def get_probe_letter_group_from_path(probe_string: str | pathlib.Path, length: Optional[int] = None) -> Literal['A', 'B', 'C', 'D', 'E', 'F']:
    """Extract the group of probe letters that follow `_probe`.
    
    Expected number of probe letters in the group may also be specified.
    
    - if multiple conflicting groups are found, ValueError is raised.
    - if multiple conflicting probe letters are found, ValueError is raised.
    
    >>> get_probe_letter_group_from_path('1184386053/1176693482_probeA')
    'A'
    >>> get_probe_letter_group_from_path(pathlib.Path('1184386053/1176693482_probeA'))
    'A'
    >>> get_probe_letter_group_from_path('1184386053/1176693482_probeABC')
    'ABC'
    >>> get_probe_letter_group_from_path('1184386053/1176693482')
    Traceback (most recent call last):
     ...
    ValueError: No probe letter found: 1184386053/1176693482
    >>> get_probe_letter_group_from_path('1176693482_probeABC/1176693482_probeA')
    Traceback (most recent call last):
     ...
    ValueError: Multiple probe letters found: 1176693482_probeABC/1176693482_probeA ('ABC', 'A')
    >>> get_probe_letter_group_from_path('1176693482_probeABC', length=1)
    Traceback (most recent call last):
     ...
    ValueError: 1 probe letter was requested, found 3: 1176693482_probeABC -> ABC
    """
    matches = get_all_probe_letter_groups_from_path(probe_string)
    if not matches:
        raise ValueError(f'No probe letter found: {probe_string}')
    if len(matches) > 1 and not all(match == matches[0] for match in matches):
        raise ValueError(f'Multiple probe letters found: {probe_string} {matches}')
    if length is not None and len(matches[0]) != length:
        raise ValueError(f'{length} probe letter{"s were" if length > 1 else " was"} requested, found {len(matches[0])}: {probe_string} -> {matches[0]}')
    return matches[0]

get_single_probe_letter_from_path = functools.partial(get_probe_letter_group_from_path, length=1)
get_three_probe_letters_from_path = functools.partial(get_probe_letter_group_from_path, length=3)

if __name__ == '__main__':
    
    logger = np_logging.getLogger()

    TEST = 1
    if TEST == 1: # doctests
        doctest.testmod(verbose=True)
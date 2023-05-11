from __future__ import annotations

import functools
import json
from typing import Any

import np_session
import numpy as np
import pandas as pd

from np_pipeline_qc.legacy import analysis, behavior_analysis
from np_pipeline_qc.legacy import probeSync_qc as probeSync
from np_pipeline_qc.legacy.get_RFs_standalone import get_RFs
from np_pipeline_qc.legacy.query_lims import query_lims
from np_pipeline_qc.legacy.sync_dataset import Dataset as sync_dataset
from np_pipeline_qc.legacy.task1_behavior_session import DocData


class BaseQC:

    default_probes_to_run = 'ABCDEF'

    cortical_sort = False
    modules_to_run = 'all'
    ctx_units_percentile = 50

    def __init__(self, session: int | str | np_session.Session) -> None:
        if not isinstance(session, np_session.Session):
            session = np_session.Session(session)
        self.session = session
        self.set_legacy_parameters()

    def run(self):
        self._run_modules()

    def set_legacy_parameters(self):
        self.errors = []

        self.data_stream_status = {
            'pkl': [False, self._load_pkl_data],
            'opto': [False, self._load_opto_data],
            'sync': [False, self._load_sync_data],
            'unit': [False, self._build_unit_table],
            'LFP': [False, self._build_lfp_dict],
        }

        self.paths: dict[
            str, Any
        ] = self.session.data_dict or np_session.local_data_getter(
            self.session.folder, cortical_sort=self.cortical_sort
        )

        self.FIG_SAVE_DIR = self.session.qc_path
        self.FIG_SAVE_DIR.mkdir(parents=True, exist_ok=True)

        self.figure_prefix = (
            self.paths['external_specimen_name']
            + '_'
            + self.paths['datestring']
            + '_'
        )

        ### GET FILE PATHS TO SYNC AND PKL FILES ###
        self.SYNC_FILE = self.paths.get('sync_file', 'none found')
        self.BEHAVIOR_PKL = self.paths.get('behavior_pkl', 'none found')
        self.REPLAY_PKL = self.paths.get('replay_pkl', 'none found')
        self.MAPPING_PKL = self.paths.get('mapping_pkl', 'none found')
        self.OPTO_PKL = self.paths.get('opto_pkl', 'none found')

        for f, s in zip(
            [
                self.SYNC_FILE,
                self.BEHAVIOR_PKL,
                self.REPLAY_PKL,
                self.MAPPING_PKL,
            ],
            ['sync: ', 'behavior: ', 'replay: ', 'mapping: '],
        ):
            print(s + f)

        self.probe_dirs = [
            self.paths['probe' + pid] for pid in self.paths['data_probes']
        ]
        self.lfp_dirs = [
            self.paths['lfp' + pid] for pid in self.paths['data_probes']
        ]

        self.probe_dict = None
        self.lfp_dict = None
        self.metrics_dict = None
        self.probeinfo_dict = None
        self.agar_channel_dict = None

    def _module_validation_decorator(data_streams):
        """Decorator to handle calling the module functions below and supplying
        the right data streams.
        INPUT:
            data_streams: This should be a list of the data streams required
                by this module function. Options are (as of 10/30/2020):
                    'pkl' : all the pkl files
                    'sync': syncdataset
                    'unit': kilosort data, builds unit table
                    'LFP': LFP data, builds lfp table
        """

        def decorator(module_func):
            def wrapper(self, **kwargs):
                for d in data_streams:
                    if not self.data_stream_status[d][0]:
                        self.data_stream_status[d][1]()
                module_func(self, **kwargs)

            return wrapper

        return decorator

    def _load_sync_data(self):
        self.syncDataset = sync_dataset(self.SYNC_FILE)

        try:
            # vr, self.vf = probeSync.get_sync_line_data(self.syncDataset, channel=2)
            self.vf = probeSync.partition_vsyncs(self.syncDataset)
            MONITOR_LAG = analysis.get_monitor_lag(self.syncDataset)
            if MONITOR_LAG > 0.06:
                self.errors.append(
                    (
                        'vsync',
                        'abnormal monitor lag {}, using default {}'.format(
                            MONITOR_LAG, 0.036
                        ),
                    )
                )
                MONITOR_LAG = 0.036

            # self.FRAME_APPEAR_TIMES = self.vf + MONITOR_LAG
            self.FRAME_APPEAR_TIMES = probeSync.get_experiment_frame_times(
                self.syncDataset
            )   # trying new vsync method
            largest_monitor_lag = 0.5   # VERY LIBERAL BOUND, will only catch things that are catastrophically bad
            if (
                np.mean(np.abs(self.FRAME_APPEAR_TIMES - self.vf))
                > largest_monitor_lag
            ):
                warning = 'Unexpected discrepancy between computed frame times and vsyncs, using raw vsyncs plus MONITOR_LAG'
                self.errors.append(('vsync', warning))
                print(warning)
                self.FRAME_APPEAR_TIMES = self.vf + MONITOR_LAG

            self.MONITOR_LAG = MONITOR_LAG
            self.vsync_times = np.copy(self.vf)
        except:
            print('error getting vsync times')
        self.data_stream_status['sync'][0] = True

    def _load_opto_data(self):

        self.opto_data = pd.read_pickle(self.OPTO_PKL)
        self.data_stream_status['opto'][0] = True

    def _load_pkl_data(self):
        if not self.data_stream_status['sync'][0]:
            self._load_sync_data()

        self.behavior_data = (
            pd.read_pickle(self.BEHAVIOR_PKL)
            if not self.BEHAVIOR_PKL == 'none found'
            else None
        )
        self.mapping_data = (
            pd.read_pickle(self.MAPPING_PKL)
            if not self.MAPPING_PKL == 'none found'
            else None
        )
        self.replay_data = (
            pd.read_pickle(self.REPLAY_PKL)
            if not self.REPLAY_PKL == 'none found'
            else None
        )
        # self.opto_data = pd.read_pickle(self.OPTO_PKL)

        self.trials = (
            behavior_analysis.get_trials_df(self.behavior_data)
            if self.behavior_data
            else None
        )

        ### CHECK FRAME COUNTS ###

        self.behavior_frame_count = (
            self.behavior_data['items']['behavior']['intervalsms'].size + 1
            if self.behavior_data
            else 0
        )
        self.mapping_frame_count = (
            self.mapping_data['intervalsms'].size + 1
            if self.mapping_data
            else 0
        )
        self.replay_frame_count = (
            self.replay_data['intervalsms'].size + 1 if self.replay_data else 0
        )

        self.total_pkl_frames = (
            self.behavior_frame_count
            + self.mapping_frame_count
            + self.replay_frame_count
        )

        ### CHECK THAT NO FRAMES WERE DROPPED FROM SYNC ###
        print('frames in pkl files: {}'.format(self.total_pkl_frames))
        print('frames in sync file: {}'.format(len(self.vf)))

        try:
            assert self.total_pkl_frames == len(self.vf)
        except AssertionError:
            print('Mismatch between sync frames and pkl frames')
            if len(self.vf) < self.total_pkl_frames:
                print(
                    'Found fewer vsyncs than pkl frames, attempting to interpolate'
                )
                new_vsyncs = probeSync.patch_vsyncs(
                    self.syncDataset,
                    self.behavior_data,
                    self.mapping_data,
                    self.replay_data,
                )
                self.vsync_times = new_vsyncs
                self.FRAME_APPEAR_TIMES = self.vsync_times + self.MONITOR_LAG

        ### CHECK THAT REPLAY AND BEHAVIOR HAVE SAME FRAME COUNT ###
        print('frames in behavior stim: {}'.format(self.behavior_frame_count))
        print('frames in replay stim: {}'.format(self.replay_frame_count))

        # assert(behavior_frame_count==replay_frame_count)

        # look for potential frame offsets from aborted stims
        (
            self.behavior_start_frame,
            self.mapping_start_frame,
            self.replay_start_frame,
        ) = probeSync.get_frame_offsets(
            self.syncDataset,
            [
                self.behavior_frame_count,
                self.mapping_frame_count,
                self.replay_frame_count,
            ],
        )
        self.behavior_end_frame = (
            self.behavior_start_frame + self.behavior_frame_count - 1
        )
        self.mapping_end_frame = (
            self.mapping_start_frame + self.mapping_frame_count - 1
        )
        self.replay_end_frame = (
            self.replay_start_frame + self.replay_frame_count - 1
        )

        (
            self.behavior_start_time,
            self.mapping_start_time,
            self.replay_start_time,
        ) = [
            self.FRAME_APPEAR_TIMES[f]
            for f in [
                self.behavior_start_frame,
                self.mapping_start_frame,
                self.replay_start_frame,
            ]
        ]
        self.behavior_end_time, self.mapping_end_time, self.replay_end_time = [
            self.FRAME_APPEAR_TIMES[f]
            for f in [
                self.behavior_end_frame,
                self.mapping_end_frame,
                self.replay_end_frame,
            ]
        ]
        self.data_stream_status['pkl'][0] = True

    def _run_modules(self):

        module_list = [
            func for func in dir(self) if callable(getattr(self, func))
        ]
        for module in module_list:
            if module[0] == '_':
                continue

            if module in self.modules_to_run or self.modules_to_run == 'all':
                func = getattr(self, module)
                print('\n' + '#' * 20)
                print('Running module: {}\n'.format(module))
                try:
                    func()
                except Exception as e:
                    print('Error running module {}'.format(module))
                    self.errors.append((module, e))

    def _build_unit_table(self):
        ### BUILD UNIT TABLE ####
        probe_df = probeSync.build_unit_table(
            self.probes_to_run, self.paths, self.syncDataset
        )
        self.probe_dict = {}
        for p in probe_df['probe'].unique():
            self.probe_dict[p] = probe_df.loc[probe_df['probe'] == p]

        self.data_stream_status['unit'][0] = True

    def _build_lfp_dict(self):
        self.lfp_dict = probeSync.build_lfp_dict(
            self.lfp_dirs, self.syncDataset
        )
        self.data_stream_status['LFP'][0] = True

    @functools.cached_property
    def metrics_dict(self) -> dict:
        return {
            p: pd.read_csv(self.paths[f'probe{p}_metrics'])
            for p in self.probes_to_run
            if f'probe{p}_metrics' in self.paths
        }

    @functools.cached_property
    def probeinfo_dict(self) -> dict:
        return {
            p: json.loads(self.paths[f'probe{p}_info'].read_bytes())
            for p in self.probes_to_run
            if f'probe{p}_info' in self.paths
        }

    @property
    def probes_to_run(self) -> str:
        if probes := ''.join(self.session.data_dict.get('data_probes', '')):
            return probes
        # TODO work out from filesystem
        return 'ABCDEF'

    @property
    def genotype(self) -> str:
        return '-'.join(
            self.session.mouse.lims.get('name', '').split('-')[:-1]
        )



# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 13:08:45 2020

@author: svc_ccg
"""
from __future__ import annotations

import datetime
import glob
import json
import logging
import os
import pathlib
import shutil
import traceback
from typing import Literal, Sequence

import np_config
import np_logging
import np_pipeline_qc.legacy.probeSync_qc as probeSync
import np_pipeline_qc.reports as reports
import np_session
import numpy as np
import pandas as pd
from np_pipeline_qc.legacy import analysis, behavior_analysis, data_getters
from np_pipeline_qc.legacy.get_RFs_standalone import get_RFs
from np_pipeline_qc.legacy.query_lims import query_lims
from np_pipeline_qc.legacy.sync_dataset import Dataset as sync_dataset
from np_pipeline_qc.legacy.task1_behavior_session import DocData
from np_pipeline_qc.sorted import spike_depths

logger = np_logging.getLogger(__name__)


class run_qc:

    modules_to_run: Literal["all"] | Sequence[str] = "all"
    cortical_sort = False
    probes_to_run = "ABCDEF"
    ctx_units_percentile = 50
    debug = False
    """Raise errors encountered in modules if True."""

    def __init__(
        self,
        exp_id,
        save_root,
        **kwargs,
    ):

        self.session = np_session.Session(exp_id)

        for _ in kwargs:
            setattr(self, _, kwargs[_])

        self.errors = []

        self.data_stream_status = {
            "pkl": [False, self._load_pkl_data],
            "opto": [False, self._load_opto_data],
            "sync": [False, self._load_sync_data],
            "unit": [False, self._build_unit_table],
            "LFP": [False, self._build_lfp_dict],
        }

        self.paths = {}
        identifier = exp_id
        try:
            if identifier.find("_") >= 0:
                d = data_getters.local_data_getter(
                    base_dir=identifier, cortical_sort=self.cortical_sort
                )
            else:
                d = data_getters.lims_data_getter(exp_id=identifier)
            self.paths = d.data_dict
        except KeyError:  # no platform json file
            self.paths["es_id"] = self.session.folder
            self.paths["external_specimen_name"] = str(self.session.mouse)
            self.paths["datestring"] = str(self.session.date).replace("-", "")
            self.paths["rig"] = str(self.session.rig)
            self.paths["sync_file"] = self.session.sync.as_posix()

        self.FIG_SAVE_DIR = save_root
        if not os.path.exists(self.FIG_SAVE_DIR):
            os.mkdir(self.FIG_SAVE_DIR)

        self.figure_prefix = (
            self.paths["external_specimen_name"] + "_" + self.paths["datestring"] + "_"
        )

        ### GET FILE PATHS TO SYNC AND PKL FILES ###
        self.SYNC_FILE = self.paths.get("sync_file", "none found")
        self.BEHAVIOR_PKL = self.paths.get("behavior_pkl", "none found")
        self.REPLAY_PKL = self.paths.get("replay_pkl", "none found")
        self.MAPPING_PKL = self.paths.get("mapping_pkl", "none found")
        self.OPTO_PKL = self.paths.get("opto_pkl", "none found")

        # update video paths if none found
        video_paths = {
            "RawBehaviorTrackingVideo": self.session.get_behavior_video(),
            "RawBehaviorTrackingVideoMetadata": self.session.get_behavior_video_info(),
            "RawEyeTrackingVideo": self.session.get_eye_video(),
            "RawEyeTrackingVideoMetadata": self.session.get_eye_video_info(),
            "RawFaceTrackingVideo": self.session.get_face_video(),
            "RawFaceTrackingVideoMetadata": self.session.get_face_video_info(),
        }
        for k, v in video_paths.items():
            if k not in self.paths:
                self.paths[k] = v.as_posix()

        for f, s in zip(
            [
                self.SYNC_FILE,
                self.BEHAVIOR_PKL,
                self.REPLAY_PKL,
                self.MAPPING_PKL,
            ],
            ["sync: ", "behavior: ", "replay: ", "mapping: "],
        ):
            print(s + f)

        try:
            self.probe_dirs = [
                self.paths["probe" + pid] for pid in self.paths["data_probes"]
            ]
        except KeyError:
            self.probe_dirs = []
        try:
            self.lfp_dirs = [
                self.paths["lfp" + pid] for pid in self.paths["data_probes"]
            ]
        except KeyError:
            self.lfp_dirs = []

        self.probe_dict = None
        self.lfp_dict = None
        self.metrics_dict = None
        self.probeinfo_dict = None
        self.agar_channel_dict = None

        self._get_genotype()
        self._get_platform_info()
        #        self._make_specimen_meta_json()
        #        self._make_session_meta_json()

        try:
            self.probes_to_run = [
                p
                for p in self.probes_to_run
                if p in self.paths["data_probes"]
                and self.paths.get(f"probe{p}_metrics")
            ]
        except KeyError:
            self.probes_to_run = []

        self._run_modules()
        self._generate_report()
        self._email_notify_recent_session()

    def _generate_report(self):
        if self.errors:
            analysis.save_json(
                {module: error for module, error in self.errors},
                os.path.join(self.FIG_SAVE_DIR, "qc_module_errors.json"),
            )
        reports.session_qc_dir_to_img_html(self.FIG_SAVE_DIR)

    def _email_notify_recent_session(self):
        """Send an email with a link to the session QC report if the session
        happened within the prev 24 hrs."""
        if (
            self.session.date >= datetime.date.today() - datetime.timedelta(days=1)
            and str(self.session.mouse) != "366122"
        ):
            if "np-exp" in self.session.npexp_path.as_posix():
                # pipeline sessions
                email = np_logging.email(
                    [
                        f"{n}@alleninstitute.org"
                        for n in np_config.fetch("/projects/np_pipeline_qc")[
                            "email_users"
                        ]
                    ],
                    exception_only=True,
                )
            else:
                email = np_logging.email(
                    [f"{n}@alleninstitute.org" for n in ("ben.hardcastle", "corbettb")],
                    exception_only=True,
                )

            email.info(
                f'QC report | {self.session} | {"Hab" if self.session.is_hab else "Ephys"} | {self.session.project}\nfile:{self.session.qc_path / "qc.html"}'
            )

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
                        "vsync",
                        "abnormal monitor lag {}, using default {}".format(
                            MONITOR_LAG, 0.036
                        ),
                    )
                )
                MONITOR_LAG = 0.036

            # self.FRAME_APPEAR_TIMES = self.vf + MONITOR_LAG
            self.FRAME_APPEAR_TIMES = probeSync.get_experiment_frame_times(
                self.syncDataset
            )  # trying new vsync method
            largest_monitor_lag = 0.5  # VERY LIBERAL BOUND, will only catch things that are catastrophically bad
            if np.mean(np.abs(self.FRAME_APPEAR_TIMES - self.vf)) > largest_monitor_lag:
                warning = "Unexpected discrepancy between computed frame times and vsyncs, using raw vsyncs plus MONITOR_LAG"
                self.errors.append(("vsync", warning))
                print(warning)
                self.FRAME_APPEAR_TIMES = self.vf + MONITOR_LAG

            self.MONITOR_LAG = MONITOR_LAG
            self.vsync_times = np.copy(self.vf)
        except:
            print("error getting vsync times")
        self.data_stream_status["sync"][0] = True

    def _load_opto_data(self):

        self.opto_data = pd.read_pickle(self.OPTO_PKL)
        self.data_stream_status["opto"][0] = True

    def _load_pkl_data(self):
        if not self.data_stream_status["sync"][0]:
            self._load_sync_data()

        self.behavior_data = (
            pd.read_pickle(self.BEHAVIOR_PKL)
            if not self.BEHAVIOR_PKL == "none found"
            else None
        )

        self.mapping_data = (
            pd.read_pickle(self.BEHAVIOR_PKL)
            if not self.MAPPING_PKL == "none found"
            else None
        )
        self.replay_data = (
            pd.read_pickle(self.REPLAY_PKL)
            if not self.REPLAY_PKL == "none found"
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
            self.behavior_data["items"]["behavior"]["intervalsms"].size + 1
            if self.behavior_data
            else 0
        )
        self.mapping_frame_count = (
            self.mapping_data["intervalsms"].size + 1 if self.mapping_data else 0
        )
        self.replay_frame_count = (
            self.replay_data["intervalsms"].size + 1 if self.replay_data else 0
        )

        self.total_pkl_frames = (
            self.behavior_frame_count
            + self.mapping_frame_count
            + self.replay_frame_count
        )

        ### CHECK THAT NO FRAMES WERE DROPPED FROM SYNC ###
        print("frames in pkl files: {}".format(self.total_pkl_frames))
        print("frames in sync file: {}".format(len(self.vf)))

        try:
            assert self.total_pkl_frames == len(self.vf)
        except AssertionError:
            print("Mismatch between sync frames and pkl frames")
            if len(self.vf) < self.total_pkl_frames:
                print("Found fewer vsyncs than pkl frames, attempting to interpolate")
                new_vsyncs = probeSync.patch_vsyncs(
                    self.syncDataset,
                    self.behavior_data,
                    self.mapping_data,
                    self.replay_data,
                )
                self.vsync_times = new_vsyncs
                self.FRAME_APPEAR_TIMES = self.vsync_times + self.MONITOR_LAG

        ### CHECK THAT REPLAY AND BEHAVIOR HAVE SAME FRAME COUNT ###
        print("frames in behavior stim: {}".format(self.behavior_frame_count))
        print("frames in replay stim: {}".format(self.replay_frame_count))

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
        self.mapping_end_frame = self.mapping_start_frame + self.mapping_frame_count - 1
        self.replay_end_frame = self.replay_start_frame + self.replay_frame_count - 1

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
        self.data_stream_status["pkl"][0] = True

    def _run_modules(self):

        module_list = [func for func in dir(self) if callable(getattr(self, func))]
        for module in module_list:
            if module[0] == "_":
                continue

            if module in self.modules_to_run or self.modules_to_run == "all":
                func = getattr(self, module)
                print("\n" + "#" * 20)
                print("Running module: {}\n".format(module))
                try:
                    func()
                except Exception as e:
                    logger.exception("Error running module {}".format(module))
                    self.errors.append((module, traceback.format_exc()))

    def _build_unit_table(self):
        ### BUILD UNIT TABLE ####
        probe_df = probeSync.build_unit_table(
            self.probes_to_run, self.paths, self.syncDataset
        )
        self.probe_dict = {}
        for p in probe_df["probe"].unique():
            self.probe_dict[p] = probe_df.loc[probe_df["probe"] == p]

        self.data_stream_status["unit"][0] = True

    def _build_lfp_dict(self):
        self.lfp_dict = probeSync.build_lfp_dict(self.lfp_dirs, self.syncDataset)
        self.data_stream_status["LFP"][0] = True

    def _build_metrics_dict(self):

        self.metrics_dict = {}
        for p in self.probes_to_run:
            key = "probe" + p + "_metrics"
            if key in self.paths:
                metrics_file = self.paths[key]
                self.metrics_dict[p] = pd.read_csv(metrics_file)

    def _build_probeinfo_dict(self):

        # read probe info json
        self.probeinfo_dict = {}
        for p in self.probes_to_run:
            key = "probe" + p + "_info"
            if key in self.paths:
                with open(self.paths[key], "r") as file:
                    self.probeinfo_dict[p] = json.load(file)

    def _get_genotype(self):
        query_string = """
            SELECT es.id as es_id, sp.name as specimen_name
            FROM ecephys_sessions es
            JOIN specimens sp ON sp.id = es.specimen_id
            WHERE es.id = {}
            ORDER BY es.id
            """
        try:
            genotype_info = query_lims(query_string.format(self.paths["es_id"]))
            if len(genotype_info) > 0 and "specimen_name" in genotype_info[0]:
                genotype_string = genotype_info[0]["specimen_name"]
                self.genotype = genotype_string[: genotype_string.rfind("-")]
            else:
                print(
                    "Could not find genotype for mouse {}".format(
                        self.paths["external_specimen_name"]
                    )
                )
                self.genotype = ""
        except Exception as e:
            self.genotype = ""
            print("Error retrieving genotype: {}".format(e))

    def _get_platform_info(self):

        # read in platform json
        try:
            platform_file = self.paths["EcephysPlatformFile"]
            with open(platform_file, "r") as file:
                self.platform_info = json.load(file)
        except Exception as e:
            print("Error getting platform json: {}".format(e))

    def _get_agar_channels(self):
        if self.probeinfo_dict is None:
            self._build_probeinfo_dict()

        self.agar_channel_dict = {}
        for pid in self.probes_to_run:
            self.agar_channel_dict[pid] = analysis.find_agar_channels(
                self.probeinfo_dict[pid]
            )

    def make_specimen_meta_json(self):

        try:
            meta = {}
            meta["mid"] = self.paths["external_specimen_name"]

            if self.genotype is None:
                self._get_genotype()

            meta["genotype"] = self.genotype
            analysis.save_json(
                meta, os.path.join(self.FIG_SAVE_DIR, "specimen_meta.json")
            )
        except Exception as e:
            print("Error making specimen meta json {}".format(e))

    def make_session_meta_json(self):
        if not hasattr(self, "behavior_data"):
            self._load_pkl_data()

        try:
            meta = {}
            meta["image_set"] = self.behavior_data["items"]["behavior"]["params"][
                "stimulus"
            ]["params"]["image_set"]
            meta["stage"] = self.behavior_data["items"]["behavior"]["params"]["stage"]
            meta["operator"] = self.behavior_data["items"]["behavior"]["params"][
                "user_id"
            ]
            meta["rig"] = self.platform_info["rig_id"]

            analysis.save_json(
                meta, os.path.join(self.FIG_SAVE_DIR, "session_meta.json")
            )
        except Exception as e:
            print("Error making session meta json {}".format(e))

    @_module_validation_decorator(data_streams=["pkl", "sync"])
    def behavior(self):
        ### Behavior Analysis ###
        behavior_plot_dir = os.path.join(self.FIG_SAVE_DIR, "behavior")
        behavior_analysis.plot_behavior(
            self.trials, behavior_plot_dir, prefix=self.figure_prefix
        )
        behavior_analysis.plot_trial_licks(
            self.trials,
            self.vsync_times,
            self.behavior_start_frame,
            behavior_plot_dir,
            prefix=self.figure_prefix,
        )
        trial_types, counts = behavior_analysis.get_trial_counts(self.trials)
        behavior_analysis.plot_trial_type_pie(
            counts, trial_types, behavior_plot_dir, prefix=self.figure_prefix
        )
        pkl_list = [
            getattr(self, pd)
            for pd in ["behavior_data", "mapping_data", "replay_data"]
            if hasattr(self, pd)
        ]
        analysis.plot_running_wheel(
            pkl_list, behavior_plot_dir, prefix=self.figure_prefix
        )

    @_module_validation_decorator(data_streams=["sync", "pkl"])
    def vsyncs(self):
        ### Plot vsync info ###
        vsync_save_dir = os.path.join(self.FIG_SAVE_DIR, "vsyncs")
        analysis.plot_frame_intervals(
            self.vf,
            self.behavior_frame_count,
            self.mapping_frame_count,
            self.behavior_start_frame,
            self.mapping_start_frame,
            self.replay_start_frame,
            vsync_save_dir,
            prefix=self.figure_prefix,
        )
        analysis.plot_vsync_interval_histogram(
            self.vf, vsync_save_dir, prefix=self.figure_prefix
        )
        analysis.vsync_report(
            self.syncDataset,
            self.total_pkl_frames,
            vsync_save_dir,
            prefix=self.figure_prefix,
        )
        analysis.plot_vsync_and_diode(
            self.syncDataset, vsync_save_dir, prefix=self.figure_prefix
        )

    def probe_noise(self, data_chunk_size=1):
        if self.probeinfo_dict is None:
            self._build_probeinfo_dict()

        noise_dir = os.path.join(self.FIG_SAVE_DIR, "probe_noise")
        analysis.plot_AP_band_noise(
            self.probe_dirs,
            self.probes_to_run,
            self.probeinfo_dict,
            noise_dir,
            data_chunk_size=data_chunk_size,
            prefix=self.figure_prefix,
        )

    def probe_yield(self):
        ### Plot Probe Yield QC ###
        if self.metrics_dict is None:
            self._build_metrics_dict()
        if self.probeinfo_dict is None:
            self._build_probeinfo_dict()

        probe_yield_dir = os.path.join(self.FIG_SAVE_DIR, "probe_yield")
        if not os.path.exists(probe_yield_dir):
            os.mkdir(probe_yield_dir)

        analysis.plot_unit_quality_hist(
            self.metrics_dict, probe_yield_dir, prefix=self.figure_prefix
        )
        analysis.plot_unit_distribution_along_probe(
            self.metrics_dict,
            self.probeinfo_dict,
            self.paths,
            probe_yield_dir,
            prefix=r"unit_distribution\\" + self.figure_prefix,
        )
        analysis.copy_probe_depth_images(
            self.paths,
            probe_yield_dir,
            prefix=r"probe_depth\\" + self.figure_prefix,
        )
        analysis.probe_yield_report(
            self.metrics_dict,
            self.probeinfo_dict,
            probe_yield_dir,
            prefix=self.figure_prefix,
        )

    @_module_validation_decorator(data_streams=["sync", "unit"])
    def data_loss(self, return_hist=False):

        probe_yield_dir = os.path.join(self.FIG_SAVE_DIR, "probe_yield")
        if not os.path.exists(probe_yield_dir):
            os.mkdir(probe_yield_dir)

        ### Look for gaps in data acquisition ###
        all_spike_hist = analysis.plot_all_spike_hist(
            self.probe_dict,
            probe_yield_dir,
            prefix=r"all_spike_hist\\" + self.figure_prefix + "good",
            return_hist=return_hist,
        )
        self.all_spike_hists = all_spike_hist

    def unit_metrics(self):
        ### Unit Metrics ###
        unit_metrics_dir = os.path.join(self.FIG_SAVE_DIR, "unit_metrics")
        analysis.plot_unit_metrics(
            self.paths, unit_metrics_dir, prefix=self.figure_prefix
        )

    def spike_depths(self):
        ### Spike Depths ###
        spike_depths_dir = os.path.join(
            self.FIG_SAVE_DIR, "probe_yield", "unit_distribution"
        )
        spike_depths.save_spike_depth_map_all_probes(
            self.session.npexp_path,
            spike_depths_dir,
            prefix=self.figure_prefix,
        )

    @_module_validation_decorator(data_streams=["sync"])
    def probeSyncAlignment(self):
        ### Probe/Sync alignment
        probeSyncDir = os.path.join(self.FIG_SAVE_DIR, "probeSyncAlignment")
        # analysis.plot_barcode_interval_hist(self.probe_dirs, self.syncDataset, probeSyncDir, prefix=self.figure_prefix)
        analysis.plot_barcode_intervals(
            self.probe_dirs,
            self.syncDataset,
            probeSyncDir,
            prefix=self.figure_prefix,
        )
        analysis.probe_sync_report(
            self.probe_dirs,
            self.syncDataset,
            probeSyncDir,
            prefix=self.figure_prefix,
        )
        analysis.plot_barcode_matches(
            self.probe_dirs,
            self.syncDataset,
            probeSyncDir,
            prefix=self.figure_prefix,
        )

    @_module_validation_decorator(data_streams=["pkl", "sync", "unit"])
    def receptive_fields(self, save_rf_mat=False, stimulus_index=0):
        ### Plot receptive fields
        if self.probe_dict is None:
            self._build_unit_table()

        if self.cortical_sort:
            ctx_units_percentile = 100
        else:
            ctx_units_percentile = self.ctx_units_percentile
        # ctx_units_percentile = 40 if not self.cortical_sort else 100
        print("built unit table, making RFs now")
        get_RFs(
            self.probe_dict,
            self.mapping_data,
            self.mapping_start_frame,
            self.FRAME_APPEAR_TIMES,
            os.path.join(self.FIG_SAVE_DIR, "receptive_fields"),
            ctx_units_percentile=ctx_units_percentile,
            prefix=self.figure_prefix,
            save_rf_mat=save_rf_mat,
            stimulus_index=stimulus_index,
        )

    @_module_validation_decorator(data_streams=["pkl", "sync", "unit"])
    def change_response(self):
        if self.probe_dict is None:
            self._build_unit_table()
        change_frames = np.array(self.trials["change_frame"].dropna()).astype(int) + 1
        analysis.plot_population_change_response(
            self.probe_dict,
            self.behavior_start_frame,
            self.replay_start_frame,
            change_frames,
            self.FRAME_APPEAR_TIMES,
            os.path.join(self.FIG_SAVE_DIR, "change_response"),
            ctx_units_percentile=self.ctx_units_percentile,
            prefix=self.figure_prefix,
        )

    @_module_validation_decorator(data_streams=["pkl", "sync", "LFP"])
    def LFP(
        self,
        agarChRange=None,
        num_licks=20,
        windowBefore=0.5,
        windowAfter=1.5,
        min_inter_lick_time=0.5,
        behavior_duration=3600,
    ):

        ### LFP ###
        self._get_agar_channels()  # to re-reference
        if self.lfp_dict is None:
            self._build_lfp_dict()
        lfp_save_dir = os.path.join(self.FIG_SAVE_DIR, "LFP")
        lick_times = analysis.get_rewarded_lick_times(
            probeSync.get_lick_times(self.syncDataset),
            self.FRAME_APPEAR_TIMES[self.behavior_start_frame :],
            self.trials,
            min_inter_lick_time=min_inter_lick_time,
        )
        analysis.plot_lick_triggered_LFP(
            self.lfp_dict,
            self.agar_channel_dict,
            lick_times,
            lfp_save_dir,
            prefix=r"lick_triggered_average\\" + self.figure_prefix,
            agarChRange=agarChRange,
            num_licks=num_licks,
            windowBefore=windowBefore,
            windowAfter=windowAfter,
            min_inter_lick_time=min_inter_lick_time,
            behavior_duration=behavior_duration,
        )

    def probe_targeting(self):

        targeting_dir = os.path.join(self.FIG_SAVE_DIR, "probe_targeting")
        # images_to_copy = ['insertion_location_image', 'overlay_image']
        images_to_copy = [
            "EcephysInsertionLocationImage",
            "EcephysOverlayImage",
        ]
        analysis.copy_files(images_to_copy, self.paths, targeting_dir)
        self.probe_insertion_report = analysis.probe_insertion_report(
            self.paths["NewstepConfiguration"],
            self.platform_info["ProbeInsertionStartTime"],
            self.platform_info["ExperimentStartTime"],
            targeting_dir,
            prefix=self.figure_prefix,
        )

    def brain_health(self):

        brain_health_dir = os.path.join(self.FIG_SAVE_DIR, "brain_health")
        images_to_copy = [
            "brain_surface_image_left",
            "pre_insertion_surface_image_left",
            "post_insertion_surface_image_left",
            "post_stimulus_surface_image_left",
            "post_experiment_surface_image_left",
        ]

        images_to_copy = [
            "EcephysBrainSurfaceLeft",
            "EcephysPreInsertionLeft",
            "EcephysPostInsertionLeft",
            "EcephysPostStimulusLeft",
            "EcephysPostExperimentLeft",
        ]

        analysis.copy_images(
            images_to_copy,
            self.paths,
            brain_health_dir,
            x_downsample_factor=0.5,
            y_downsample_factor=0.5,
        )

    @_module_validation_decorator(data_streams=["sync"])
    def videos(self, frames_for_each_epoch=[2, 2, 2]):
        ### VIDEOS ###
        video_dir = os.path.join(self.FIG_SAVE_DIR, "videos")
        analysis.lost_camera_frame_report(
            self.paths, video_dir, prefix=self.figure_prefix
        )
        analysis.camera_frame_grabs_simple(
            self.paths,
            video_dir,
            prefix=self.figure_prefix,
        )

    @_module_validation_decorator(data_streams=["sync", "opto", "unit"])
    def optotagging(self, **kwargs):
        ### Plot opto responses along probe ###
        opto_dir = os.path.join(self.FIG_SAVE_DIR, "optotagging")
        if self.probe_dict is None:
            self._build_unit_table()

        analysis.plot_opto_responses(
            self.probe_dict,
            self.opto_data,
            self.syncDataset,
            opto_dir,
            prefix=self.figure_prefix,
            opto_sample_rate=10000,
            **kwargs,
        )


class run_qc_hab(run_qc):
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
            def wrapper(self):
                for d in data_streams:
                    if not self.data_stream_status[d][0]:
                        self.data_stream_status[d][1]()
                module_func(self)

            return wrapper

        return decorator

    def _load_pkl_data(self):  
        if not self.data_stream_status["sync"][0]:
            self._load_sync_data()

        self.behavior_data = pd.read_pickle(self.BEHAVIOR_PKL)
        self.trials = behavior_analysis.get_trials_df(self.behavior_data)

        self.mapping_data = pd.read_pickle(self.BEHAVIOR_PKL)  # MAPPING_PKL 2025

        ### CHECK FRAME COUNTS ###
        self.behavior_frame_count = (
            self.behavior_data["items"]["behavior"]["total_frames"].size ##this might be wrong since things are sandwiched now
            + 1  # prob use total frames
        )

        ##shouldn't need all his now that it's all in behavior pkl 2025
        if self.MAPPING_PKL != 'none found':
            self.mapping_data = pd.read_pickle(self.MAPPING_PKL)
            self.mapping_frame_count = self.mapping_data['intervalsms'].size + 1 ##25^
        else:
            self.mapping_frame_count = 0

        self.total_pkl_frames = (
            self.behavior_frame_count
            # + self.mapping_frame_count
            # + self.replay_frame_count ##commented out for psycode 2025
        )

        #        # look for potential frame offsets from aborted stims
        (
            self.behavior_start_frame,
            #self.mapping_start_frame, #2025
        ) = probeSync.get_frame_offsets(
            self.syncDataset,
            [self.behavior_frame_count] #, self.mapping_frame_count], #2025
        )

        self.replay_start_frame = self.total_pkl_frames
        #
        self.behavior_end_frame = (
            self.behavior_start_frame + self.behavior_frame_count - 1
        )

        self.behavior_start_time = self.FRAME_APPEAR_TIMES[self.behavior_start_frame]
        self.behavior_end_time = self.FRAME_APPEAR_TIMES[self.behavior_end_frame]

        self.data_stream_status["pkl"][0] = True

    @_module_validation_decorator(data_streams=["pkl", "sync"])
    def behavior(self):
        ### Behavior Analysis ###
        behavior_plot_dir = os.path.join(self.FIG_SAVE_DIR, "behavior")
        behavior_analysis.plot_behavior(
            self.trials, behavior_plot_dir, prefix=self.figure_prefix
        )
        behavior_analysis.plot_trial_licks(
            self.trials,
            self.vsync_times,
            self.behavior_start_frame,
            behavior_plot_dir,
            prefix=self.figure_prefix,
        )
        trial_types, counts = behavior_analysis.get_trial_counts(self.trials)
        behavior_analysis.plot_trial_type_pie(
            counts, trial_types, behavior_plot_dir, prefix=self.figure_prefix
        )
        pkl_list = [
            getattr(self, pd)
            for pd in ["behavior_data", "mapping_data", "replay_data"]
            if hasattr(self, pd)
        ]
        analysis.plot_running_wheel(
            pkl_list,
            behavior_plot_dir,
            save_plotly=False,
            prefix=self.figure_prefix,
        )

class run_qc_psycode(run_qc):
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

    def _load_pkl_data(self):
        if not self.data_stream_status["sync"][0]:
            self._load_sync_data()

        base_dir = os.path.dirname(self.SYNC_FILE)
        mapping_pkl = (
            glob.glob(os.path.join(base_dir, "*mapping.pkl"))  # added for ttn
            + glob.glob(os.path.join(base_dir, "*behavior.pkl"))
            + glob.glob(os.path.join(base_dir, "*stim.pkl"))
            + glob.glob(os.path.join(base_dir, "*PASSIVE*.pkl"))
        )
        self.MAPPING_PKL = mapping_pkl[0]  
        print("Found mapping pkl: {}".format(mapping_pkl))
        self.mapping_data = pd.read_pickle(self.MAPPING_PKL)
        print(self.mapping_data.keys())
        # self.mapping_stim_index = [
        #     istim
        #     for istim, stim in enumerate(self.mapping_data['stimuli'])
        #     if ('gabor' in stim['stim_path'])
        #     or ('receptive_field' in stim['stim_path'])
        # ]
        self.mapping_stim_index = [0]  # should work for psycode 1/31/25
        

        if len(self.mapping_stim_index) == 0:
            print("No mapping stim found in pkl file")
            return
        else:
            self.mapping_stim_index = self.mapping_stim_index[0]

        ### CHECK FRAME COUNTS ###
        pre_or_post = "post_receptive_field_mapping"#"pre_receptive_field_mapping" 
        self.mapping_frame_count =   self.mapping_data["items"]["behavior"]["items"][
            pre_or_post
        ]["static_stimulus"][
            "total_frames"
        ]  # self.mapping_data['intervalsms'].size + 1
        print('value of mapping_frant_count: "{}"'.format(self.mapping_frame_count))
        
        
        frame_rate = self.mapping_data.get("fps", 60)
        # self.mapping_start_frame = (
        #     self.mapping_data['stimuli'][
        #         'display_sequence'
        #     ][0][0]
        #     * frame_rate
        # ) 
        self.mapping_start_frame = (
             self.mapping_data["items"]["behavior"]["items"][
                pre_or_post
            ][
                "static_stimulus"
            ][  # changed from ['stimuli'][0] 2/4/2025
                "frame_list"
            ][
                0
            ]
            * frame_rate
        )
        print('value of mapping_start_frame: "{}"'.format(self.mapping_start_frame))
        self.data_stream_status["pkl"][0] = True

    @_module_validation_decorator(data_streams=["pkl", "sync", "unit"])
    def receptive_fields(
        self,
        pre_or_post,
        save_rf_mat=False,
        stimulus_index=None,
        mapping_start_frame=None,
        return_rfs=False,
    ):
        ### Plot receptive fields
        if self.probe_dict is None:
            self._build_unit_table()

        if self.cortical_sort:
            ctx_units_percentile = 100
        else:
            ctx_units_percentile = self.ctx_units_percentile
        
        if mapping_start_frame is None:
            mapping_start_frame = int(self.mapping_start_frame)
        
        if stimulus_index is None:
            stimulus_index = self.mapping_stim_index
        # ctx_units_percentile = 40 if not self.cortical_sort else 100
        print('pre or postcheck in run_qc_class rf module', pre_or_post)
        rfs = get_RFs(
            self.probe_dict,
            self.mapping_data,
            mapping_start_frame,
            self.FRAME_APPEAR_TIMES,
            os.path.join(self.FIG_SAVE_DIR, "receptive_fields"),
            pre_or_post=pre_or_post,
            return_rfs=return_rfs,
            ctx_units_percentile=ctx_units_percentile,
            prefix=self.figure_prefix,
            save_rf_mat=save_rf_mat,
            stimulus_index=self.mapping_stim_index,
        )

        self.rf_mats = rfs

    @_module_validation_decorator(data_streams=["sync"])
    def videos(self, frames_for_each_epoch=[2, 2, 2]):
        ### VIDEOS ###
        video_dir = os.path.join(self.FIG_SAVE_DIR, "videos")

        frame_times = self.FRAME_APPEAR_TIMES[:: int(self.FRAME_APPEAR_TIMES.size / 3)]
        frame_times = np.append(frame_times, self.FRAME_APPEAR_TIMES[-1])

        analysis.lost_camera_frame_report(
            self.paths, video_dir, prefix=self.figure_prefix
        )
        analysis.camera_frame_grabs_simple(
            self.paths,
            video_dir,
            prefix=self.figure_prefix,
        )

    @_module_validation_decorator(data_streams=["sync"])
    def vsyncs(self):
        ### Plot vsync info ###
        vsync_save_dir = os.path.join(self.FIG_SAVE_DIR, "vsyncs")
        analysis.plot_diode_measured_sync_square_flips(
            self.syncDataset,
            vsync_save_dir,
            prefix=self.figure_prefix,
        )
        analysis.plot_vsync_interval_histogram(
            self.vf, vsync_save_dir, prefix=self.figure_prefix
        )
        # analysis.vsync_report(self.syncDataset, self.total_pkl_frames, vsync_save_dir, prefix = self.figure_prefix)
        analysis.plot_vsync_and_diode(
            self.syncDataset, vsync_save_dir, prefix=self.figure_prefix
        )

    @_module_validation_decorator(data_streams=["pkl", "sync"])
    def behavior(self):
        ### Behavior Analysis ###
        behavior_plot_dir = os.path.join(self.FIG_SAVE_DIR, "behavior")
        
        # self.behavior_session.plotSummary(
        #     save_dir=behavior_plot_dir, prefix=self.figure_prefix
        # )
        # self.behavior_session.trial_pie(
        #     save_dir=behavior_plot_dir, prefix=self.figure_prefix
        # )
        analysis.plot_running_wheel(
            [self.mapping_data], behavior_plot_dir, prefix=self.figure_prefix
        )

    def _run_modules(self): ##edit this for psycode - add 2nd RF module for post injection RF
        no_run_list = ["LFP", "change_response"]
        module_list = [func for func in dir(self) if callable(getattr(self, func))]
        rf_count = 0
        
        module_list.append("receptive_fields")
        for module in module_list:
            if module[0] == "_" or module in no_run_list:
                continue

            if module in self.modules_to_run or self.modules_to_run == "all":
                func = getattr(self, module)
                print("\n" + "#" * 20)
                print("Running module: {}\n".format(module))
                try:
                    if module == "receptive_fields":
                        self._load_pkl_data()
                        if rf_count == 0:
                            func(stimulus_index=self.mapping_stim_index, pre_or_post='pre_receptive_field_mapping')
                            rf_count +=1
                        else:
                            func(stimulus_index=self.mapping_stim_index, pre_or_post='post_receptive_field_mapping')
                    else:
                        func()
                except Exception as e:
                    if self.debug:
                        raise e
                    logger.exception("Error running module %s", module)
                    # append full traceback
                    self.errors.append((module, traceback.format_exc()))

class run_qc_passive(run_qc):
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

    def _load_pkl_data(self):
        if not self.data_stream_status["sync"][0]:
            self._load_sync_data()

        base_dir = os.path.dirname(self.SYNC_FILE)
        mapping_pkl = (
            glob.glob(os.path.join(base_dir, "*mapping.pkl"))  # added for ttn
            + glob.glob(os.path.join(base_dir, "*behavior.pkl"))
            + glob.glob(os.path.join(base_dir, "*stim.pkl"))
            + glob.glob(os.path.join(base_dir, "*PASSIVE*.pkl"))
        )
        self.MAPPING_PKL = mapping_pkl[0]  
        print("Found mapping pkl: {}".format(mapping_pkl))
        self.mapping_data = pd.read_pickle(self.MAPPING_PKL)
        print(self.mapping_data.keys())
        # self.mapping_stim_index = [
        #     istim
        #     for istim, stim in enumerate(self.mapping_data['stimuli'])
        #     if ('gabor' in stim['stim_path'])
        #     or ('receptive_field' in stim['stim_path'])
        # ]
        self.mapping_stim_index = [0]  # should work for psycode 1/31/25
        

        if len(self.mapping_stim_index) == 0:
            print("No mapping stim found in pkl file")
            return
        else:
            self.mapping_stim_index = self.mapping_stim_index[0]

        ### CHECK FRAME COUNTS ###
        pre_or_post = "post_receptive_field_mapping"#"pre_receptive_field_mapping" 
        self.mapping_frame_count =   self.mapping_data["items"]["behavior"]["items"][
            pre_or_post
        ]["static_stimulus"][
            "total_frames"
        ]  # self.mapping_data['intervalsms'].size + 1
        print('value of mapping_frant_count: "{}"'.format(self.mapping_frame_count))
        
        
        frame_rate = self.mapping_data.get("fps", 60)
        # self.mapping_start_frame = (
        #     self.mapping_data['stimuli'][
        #         'display_sequence'
        #     ][0][0]
        #     * frame_rate
        # ) 
        self.mapping_start_frame = (
             self.mapping_data["items"]["behavior"]["items"][
                pre_or_post
            ][
                "static_stimulus"
            ][  # changed from ['stimuli'][0] 2/4/2025
                "frame_list"
            ][
                0
            ]
            * frame_rate
        )
        print('value of mapping_start_frame: "{}"'.format(self.mapping_start_frame))
        self.data_stream_status["pkl"][0] = True

    @_module_validation_decorator(data_streams=["pkl", "sync", "unit"])
    def receptive_fields(
        self,
        save_rf_mat=False,
        stimulus_index=None,
        mapping_start_frame=None,
        return_rfs=False,
    ):
        ### Plot receptive fields
        if self.probe_dict is None:
            self._build_unit_table()

        if self.cortical_sort:
            ctx_units_percentile = 100
        else:
            ctx_units_percentile = self.ctx_units_percentile
        print("got unit table")
        if mapping_start_frame is None:
            mapping_start_frame = int(self.mapping_start_frame)
        print("got starting frame")
        if stimulus_index is None:
            stimulus_index = self.mapping_stim_index
        # ctx_units_percentile = 40 if not self.cortical_sort else 100
        print("got stim index")
        rfs = get_RFs(
            self.probe_dict,
            self.mapping_data,
            mapping_start_frame,
            self.FRAME_APPEAR_TIMES,
            os.path.join(self.FIG_SAVE_DIR, "receptive_fields"),
            return_rfs=return_rfs,
            ctx_units_percentile=ctx_units_percentile,
            prefix=self.figure_prefix,
            save_rf_mat=save_rf_mat,
            stimulus_index=self.mapping_stim_index,
        )

        self.rf_mats = rfs

    @_module_validation_decorator(data_streams=["sync"])
    def videos(self, frames_for_each_epoch=[2, 2, 2]):
        ### VIDEOS ###
        video_dir = os.path.join(self.FIG_SAVE_DIR, "videos")

        frame_times = self.FRAME_APPEAR_TIMES[:: int(self.FRAME_APPEAR_TIMES.size / 3)]
        frame_times = np.append(frame_times, self.FRAME_APPEAR_TIMES[-1])

        analysis.lost_camera_frame_report(
            self.paths, video_dir, prefix=self.figure_prefix
        )
        analysis.camera_frame_grabs_simple(
            self.paths,
            video_dir,
            prefix=self.figure_prefix,
        )

    @_module_validation_decorator(data_streams=["sync"])
    def vsyncs(self):
        ### Plot vsync info ###
        vsync_save_dir = os.path.join(self.FIG_SAVE_DIR, "vsyncs")
        analysis.plot_diode_measured_sync_square_flips(
            self.syncDataset,
            vsync_save_dir,
            prefix=self.figure_prefix,
        )
        analysis.plot_vsync_interval_histogram(
            self.vf, vsync_save_dir, prefix=self.figure_prefix
        )
        # analysis.vsync_report(self.syncDataset, self.total_pkl_frames, vsync_save_dir, prefix = self.figure_prefix)
        analysis.plot_vsync_and_diode(
            self.syncDataset, vsync_save_dir, prefix=self.figure_prefix
        )

    @_module_validation_decorator(data_streams=["pkl", "sync"])
    def behavior(self):
        ### Behavior Analysis ###
        behavior_plot_dir = os.path.join(self.FIG_SAVE_DIR, "behavior")
        
        # self.behavior_session.plotSummary(
        #     save_dir=behavior_plot_dir, prefix=self.figure_prefix
        # )
        # self.behavior_session.trial_pie(
        #     save_dir=behavior_plot_dir, prefix=self.figure_prefix
        # )
        analysis.plot_running_wheel(
            [self.mapping_data], behavior_plot_dir, prefix=self.figure_prefix
        )

    def _run_modules(self): ##edit this for psycode - add 2nd RF module for post injection RF
        no_run_list = ["LFP", "change_response"]
        module_list = [func for func in dir(self) if callable(getattr(self, func))]
        print('Module list: ' + module_list)
        for module in module_list:
            if module[0] == "_" or module in no_run_list:
                continue

            if module in self.modules_to_run or self.modules_to_run == "all":
                func = getattr(self, module)
                print("\n" + "#" * 20)
                print("Running module: {}\n".format(module))
                try:
                    if module == "receptive_fields":
                        self._load_pkl_data()
                        func(stimulus_index=self.mapping_stim_index)
                    else:
                        func()
                except Exception as e:
                    if self.debug:
                        raise e
                    logger.exception("Error running module %s", module)
                    # append full traceback
                    self.errors.append((module, traceback.format_exc()))


class DR1(run_qc):
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
            def wrapper(self):
                for d in data_streams:
                    if not self.data_stream_status[d][0]:
                        self.data_stream_status[d][1]()
                module_func(self)

            return wrapper

        return decorator

    @property
    def BEHAVIOR_PKL(self) -> str:
        converted = (
            pathlib.Path(
                "//allen/programs/braintv/workgroups/nc-ephys/converted-pickles-060923"
            )
            / f"{self.session}.behavior.pkl"
        )
        if converted.exists():
            return str(converted)
        return self._behavior_pkl

    @BEHAVIOR_PKL.setter
    def BEHAVIOR_PKL(self, value: str):
        self._behavior_pkl = value

    def _load_pkl_data(self):
        if not self.data_stream_status["sync"][0]:
            self._load_sync_data()

        self.behavior_data = pd.read_pickle(self.BEHAVIOR_PKL)
        self.behavior_session = DocData(self.BEHAVIOR_PKL)
        self.behavior_session.loadBehavData()
        # self.trials = behavior_analysis.get_trials_df(self.behavior_data)

        self.mapping_data = pd.read_pickle(self.MAPPING_PKL)

        ### CHECK FRAME COUNTS ###

        self.behavior_frame_count = (
            self.behavior_data["items"]["behavior"]["intervalsms"].size + 1
        )
        self.mapping_frame_count = self.mapping_data["intervalsms"].size + 1
        self.replay_frame_count = 0  # self.replay_data['intervalsms'].size + 1

        self.total_pkl_frames = (
            self.behavior_frame_count
            + self.mapping_frame_count
            + self.replay_frame_count
        )

        #        # look for potential frame offsets from aborted stims
        (
            self.behavior_start_frame,
            self.mapping_start_frame,
        ) = probeSync.get_frame_offsets(
            self.syncDataset,
            [self.behavior_frame_count, self.mapping_frame_count],
        )

        self.replay_start_frame = self.total_pkl_frames
        #
        self.behavior_end_frame = (
            self.behavior_start_frame + self.behavior_frame_count - 1
        )

        self.behavior_start_time = self.FRAME_APPEAR_TIMES[self.behavior_start_frame]
        self.behavior_end_time = self.FRAME_APPEAR_TIMES[self.behavior_end_frame]

        self.data_stream_status["pkl"][0] = True

    @_module_validation_decorator(data_streams=["sync"])
    def videos(self, frames_for_each_epoch=[2, 2, 2]):
        ### VIDEOS ###
        video_dir = os.path.join(self.FIG_SAVE_DIR, "videos")

        frame_times = self.FRAME_APPEAR_TIMES[:: int(self.FRAME_APPEAR_TIMES.size / 3)]
        frame_times = np.append(frame_times, self.FRAME_APPEAR_TIMES[-1])

        analysis.lost_camera_frame_report(
            self.paths, video_dir, prefix=self.figure_prefix
        )
        analysis.camera_frame_grabs_simple(
            self.paths,
            video_dir,
            prefix=self.figure_prefix,
        )

    @_module_validation_decorator(data_streams=["pkl", "sync", "unit"])
    def change_response(self):
        if self.probe_dict is None:
            self._build_unit_table()

        save_dir = os.path.join(self.FIG_SAVE_DIR, "change_response")
        bs = self.behavior_session
        block_change_frames = [
            bs.changeFrames[(bs.block == bl)] for bl in np.unique(bs.block)
        ]
        analysis.plot_change_response_DR(
            self.probe_dict,
            self.behavior_start_frame,
            block_change_frames,
            self.FRAME_APPEAR_TIMES,
            save_dir,
            prefix=self.figure_prefix,
            ctx_units_percentile=66,
        )

    @_module_validation_decorator(data_streams=["pkl", "sync"])
    def behavior(self):
        ### Behavior Analysis ###
        behavior_plot_dir = os.path.join(self.FIG_SAVE_DIR, "behavior")
        self.behavior_session.plot_licks_from_change(
            save_dir=behavior_plot_dir, prefix=self.figure_prefix
        )
        self.behavior_session.plotSummary(
            save_dir=behavior_plot_dir, prefix=self.figure_prefix
        )
        self.behavior_session.trial_pie(
            save_dir=behavior_plot_dir, prefix=self.figure_prefix
        )

        pkl_list = [
            getattr(self, pd)
            for pd in ["behavior_data", "mapping_data", "replay_data"]
            if hasattr(self, pd)
        ]
        analysis.plot_running_wheel(
            pkl_list,
            behavior_plot_dir,
            save_plotly=False,
            prefix=self.figure_prefix,
        )

# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 17:20:54 2022

@author: svc_ccg
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from np_pipeline_qc.legacy.analysis import save_figure

matplotlib.rcParams['pdf.fonttype'] = 42


class DocData:
    def __init__(self, pkl_path=None):
        self.frameRate = 60
        self.behavDataPath = pkl_path

    def loadBehavData(self):
        if self.behavDataPath is None:
            self.behavDataPath = fileIO.getFile(
                'Select behavior data file', fileType='*.pkl'
            )

        if len(self.behavDataPath) == 0:
            return

        pkl = pd.read_pickle(self.behavDataPath)
        self.params = pkl['items']['behavior']['params']
        self.trialLog = np.array(pkl['items']['behavior']['trial_log'][:-1])
        #        changeLog = pkl['items']['behavior']['stimuli']['images-params']['change_log']
        #        setLog = pkl['items']['behavior']['stimuli']['images-params']['set_log']

        nTrials = len(self.trialLog)
        self.trialStartTimes = np.full(nTrials, np.nan)
        self.trialStartFrames = np.full(nTrials, np.nan)
        self.trialEndTimes = np.full(nTrials, np.nan)
        self.abortedTrials = np.zeros(nTrials, dtype=bool)
        self.abortTimes = np.full(nTrials, np.nan)
        self.changeTimes = np.full(nTrials, np.nan)
        self.changeFrames = np.full(nTrials, np.nan)
        self.changeTrials = np.zeros(nTrials, dtype=bool)
        self.catchTrials = np.zeros(nTrials, dtype=bool)
        self.warmupTrials = np.zeros(nTrials, dtype=bool)
        self.rewardTimes = np.full(nTrials, np.nan)
        self.autoReward = np.zeros(nTrials, dtype=bool)
        self.hit = np.zeros(nTrials, dtype=bool)
        self.miss = np.zeros(nTrials, dtype=bool)
        self.falseAlarm = np.zeros(nTrials, dtype=bool)
        self.correctReject = np.zeros(nTrials, dtype=bool)
        self.rewardsOn = np.zeros(nTrials, dtype=bool)
        self.block = np.zeros(nTrials)
        self.preContrast = np.full(nTrials, np.nan)
        self.postContrast = np.full(nTrials, np.nan)
        self.preImage = ['' for _ in self.trialLog]
        self.postImage = self.preImage.copy()
        self.preLabel = self.preImage.copy()
        self.postLabel = self.preImage.copy()
        current_block = 0
        for i, trial in enumerate(self.trialLog):
            events = [event[0] for event in trial['events']]
            self.rewardsOn[i] = trial['licks_enabled']
            for event, epoch, t, frame in trial['events']:
                if event == 'trial_start':
                    self.trialStartTimes[i] = t
                    self.trialStartFrames[i] = frame
                elif event == 'trial_end':
                    self.trialEndTimes[i] = t
                elif 'abort' in events:
                    if event == 'abort':
                        self.abortedTrials[i] = True
                        self.abortTimes[i] = t
                elif event in ('stimulus_changed', 'sham_change'):
                    self.changeTimes[i] = t
                    self.changeFrames[i] = frame
                elif event == 'hit':
                    self.hit[i] = True
                elif event == 'miss':
                    self.miss[i] = True
                elif event == 'false_alarm':
                    self.falseAlarm[i] = True
                elif event == 'rejection':
                    self.correctReject[i] = True
            if 'warmup' in trial['trial_params']:
                self.warmupTrials[i] = trial['trial_params']['warmup']
            if i > 0:
                if trial['licks_enabled'] == self.rewardsOn[i - 1]:
                    self.block[i] = current_block
                else:
                    current_block += 1
                    self.block[i] = current_block
            if len(trial['rewards']) > 0:
                self.rewardTimes[i] = trial['rewards'][0][1]
                self.autoReward[i] = trial['trial_params']['auto_reward']
            if not self.abortedTrials[i]:
                if trial['trial_params']['catch']:
                    self.catchTrials[i] = True
                else:
                    self.changeTrials[i] = True
                if len(trial['stimulus_changes']) > 0:
                    if trial['stimulus_changes'][0][0][0] is not None:
                        self.preImage[i] = trial['stimulus_changes'][0][0][0]
                        self.postImage[i] = trial['stimulus_changes'][0][1][0]
                        self.preContrast[i] = trial['stimulus_changes'][0][0][
                            1
                        ]['contrast']
                        self.postContrast[i] = trial['stimulus_changes'][0][1][
                            1
                        ]['contrast']
                        self.preLabel[i] = (
                            self.preImage[i]
                            + ' ('
                            + str(self.preContrast[i])
                            + ')'
                        )
                        self.postLabel[i] = (
                            self.postImage[i]
                            + ' ('
                            + str(self.postContrast[i])
                            + ')'
                        )

        frameIntervals = pkl['items']['behavior']['intervalsms'] / 1000
        frameTimes = np.concatenate(([0], np.cumsum(frameIntervals)))
        frameTimes += (
            self.trialStartTimes[0] - frameTimes[int(self.trialStartFrames[0])]
        )

        lickFrames = pkl['items']['behavior']['lick_sensors'][0]['lick_events']
        self.lickTimes = frameTimes[lickFrames]

        self.engaged = np.array(
            [
                np.sum(
                    self.hit[self.changeTrials][
                        (self.changeTimes[self.changeTrials] > t - 60)
                        & (self.changeTimes[self.changeTrials] < t + 60)
                    ]
                )
                > 1
                for t in self.changeTimes
            ]
        )

        self.labels = sorted(list(set(self.preLabel + self.postLabel))[1:])
        self.trialCount = np.zeros((len(self.labels),) * 2)
        self.trialCountEngaged = self.trialCount.copy()
        self.respCountEngaged = self.trialCount.copy()
        self.imageChange = self.trialCount.astype(bool)
        for i, postLbl in enumerate(self.labels):
            for j, preLbl in enumerate(self.labels):
                img = [lbl[: lbl.find(' (')] for lbl in (preLbl, postLbl)]
                if img[0] != img[1]:
                    self.imageChange[i, j] = True
                for pre, post, h, fa, wu, ar, eng in zip(
                    self.preLabel,
                    self.postLabel,
                    self.hit,
                    self.falseAlarm,
                    self.warmupTrials,
                    self.autoReward,
                    self.engaged,
                ):
                    if pre == preLbl and post == postLbl and not wu:
                        self.trialCount[i, j] += 1
                        if not ar and eng:
                            self.trialCountEngaged[i, j] += 1
                            self.respCountEngaged[i, j] += h or fa
        self.respRate = self.respCountEngaged / self.trialCountEngaged

    def plotSummary(self, save_dir=None, prefix=''):
        for d, lbl in zip(
            (self.trialCount, self.respRate), ('Trials', 'Response Rate')
        ):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            im = ax.imshow(d, clim=[0, d.max()], cmap='gray')
            for i, postLbl in enumerate(self.labels):
                for j, preLbl in enumerate(self.labels):
                    if not self.imageChange[i, j]:
                        ax.plot(i, j, 'rx')
            ax.set_xticks(np.arange(len(self.labels)))
            ax.set_xticklabels(self.labels, rotation=90)
            ax.set_xlabel('Pre Image (contrast)')
            ax.set_xlim([-0.5, len(self.labels) - 0.5])
            ax.set_yticks(np.arange(len(self.labels)))
            ax.set_yticklabels(self.labels)
            ax.set_ylabel('Change Image (contrast)')
            ax.set_ylim([len(self.labels) - 0.5, -0.5])
            ax.set_title(lbl + ' (x = no image identity change)')
            cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.15)
            plt.tight_layout()

            if save_dir is not None:
                save_figure(
                    fig, os.path.join(save_dir, prefix + '_task1_trialmat.png')
                )

    def trial_pie(self, save_dir=None, prefix=''):

        go_trials = np.sum(self.changeTrials)
        catch_trials = np.sum(self.catchTrials)
        autorewarded = np.sum(self.autoReward)
        aborted = np.sum(self.abortedTrials)

        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches([10, 4])
        ax[0].pie([autorewarded, catch_trials, go_trials, aborted])
        ax[0].legend(['autoreward', 'catch', 'go', 'aborted'])
        ax[0].set_title('Total trial types')

        hits = np.sum(self.hit & self.rewardsOn)
        misses = np.sum(self.miss & self.rewardsOn)
        fas = np.sum(self.falseAlarm & self.rewardsOn)
        crs = np.sum(self.correctReject & self.rewardsOn)
        ax[1].pie([crs, misses, hits, fas])
        ax[1].legend(['cr', 'miss', 'hit', 'fa'])
        ax[1].set_title('Reward on')

        hits = np.sum(self.hit & ~self.rewardsOn)
        misses = np.sum(self.miss & ~self.rewardsOn)
        fas = np.sum(self.falseAlarm & ~self.rewardsOn)
        crs = np.sum(self.correctReject & ~self.rewardsOn)
        ax[2].pie([crs, misses, hits, fas])
        ax[2].legend(['cr', 'miss', 'hit', 'fa'])
        ax[2].set_title('Reward off')

        if save_dir is not None:
            save_figure(
                fig,
                os.path.join(save_dir, prefix + '_task1_trialtypes_pie.png'),
            )

    def plot_licks_from_change(
        self, min_inter_lick_time=1, ax=None, save_dir=None, prefix=''
    ):
        # change_frames = np.array(trials['change_frame'].dropna()).astype(int)+1
        # resp_types =  ['MISS', 'HIT', 'FA', 'CR']
        if ax is None:
            fig, ax = plt.subplots()

        colors = ['orange', 'g', 'r', 'b', 'k']

        ax.axvline(0.15, c='k', linestyle='--')
        # fig.set_size_inches([6, 12])
        trial_counter = 0
        for i in range(len(self.trialLog)):
            # if self.hit[i] or self.falseAlarm[i] or self.correctReject[i] or self.miss[i]:
            if not np.isnan(self.changeTimes[i]):
                ir = np.where(
                    [
                        self.miss[i],
                        self.hit[i],
                        self.falseAlarm[i],
                        self.correctReject[i],
                        1,
                    ]
                )[0][0]
                lts = self.lickTimes[
                    (self.lickTimes >= self.trialStartTimes[i])
                    & (self.lickTimes < self.trialEndTimes[i])
                ]
                change_time = self.changeTimes[i]
                lts = lts - change_time

                if not self.rewardsOn[i]:
                    if len(lts) > 0:
                        first_trial_licks = (
                            lts[
                                np.insert(
                                    np.diff(lts) >= min_inter_lick_time,
                                    0,
                                    True,
                                )
                            ]
                            + change_time
                        )
                        if any(first_trial_licks < change_time):
                            continue
                    rect = Rectangle(
                        (-0.5, trial_counter),
                        2.5,
                        1,
                        linewidth=0,
                        edgecolor='none',
                        facecolor='0.8',
                    )
                    ax.add_patch(rect)

                ax.plot(
                    lts,
                    trial_counter * np.ones(len(lts)),
                    '|',
                    c=colors[ir],
                    ms=3,
                    markeredgewidth=3,
                )

                trial_counter += 1

        ax.set_xlim([-0.5, 2])
        ax.set_xlabel('Time from change (s)')
        ax.set_ylabel('Trial Number')

        if save_dir is not None:
            save_figure(
                fig, os.path.join(save_dir, prefix + '_task1_lickRaster.png')
            )

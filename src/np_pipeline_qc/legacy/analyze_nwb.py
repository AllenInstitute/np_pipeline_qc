# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 23:46:31 2022

@author: svc_ccg
"""
import glob
import os

import numpy as np
import pandas as pd
import scipy.signal
from allensdk.brain_observatory.ecephys.behavior_ecephys_session import (
    BehaviorEcephysSession,
)
from matplotlib import pyplot as plt
from numba import njit
from pynwb import NWBHDF5IO


@njit
def makePSTH_numba(
    spikes,
    startTimes,
    windowDur,
    binSize=0.001,
    convolution_kernel=0.05,
    avg=True,
):
    spikes = spikes.flatten()
    startTimes = startTimes - convolution_kernel / 2
    windowDur = windowDur + convolution_kernel
    bins = np.arange(0, windowDur + binSize, binSize)
    convkernel = np.ones(int(convolution_kernel / binSize))
    counts = np.zeros(bins.size - 1)
    for i, start in enumerate(startTimes):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start + windowDur)
        counts = (
            counts + np.histogram(spikes[startInd:endInd] - start, bins)[0]
        )

    counts = counts / startTimes.size
    counts = np.convolve(counts, convkernel) / (binSize * convkernel.size)
    return (
        counts[convkernel.size - 1 : -convkernel.size],
        bins[: -convkernel.size - 1],
    )


nwb_base = r'\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\vbn_s3_cache\visual-behavior-neuropixels-0.1.0\ecephys_sessions'
nwb_paths = glob.glob(os.path.join(nwb_base, '*nwb'))
nwb_validation_dir = (
    r'\\allen\programs\mindscope\workgroups\np-exp\VBN_NWB_validation'
)


def opto_validation(session):
    #    with NWBHDF5IO(nwb_path, 'r', load_namespaces=True) as nwb_io:
    #        session = BehaviorEcephysSession.from_nwb(nwbfile=nwb_io.read())
    #
    #    print('processing file {} of {}'.format(inwb, len(nwb_paths)))

    channels = session.get_channels()

    units = session.get_units()
    good_unit_filter = (
        (units['snr'] > 1)
        & (units['isi_violations'] < 1)
        & (units['firing_rate'] > 0.1)
    )
    units = units.loc[good_unit_filter]
    unitchannels = units.merge(
        channels, left_on='peak_channel_id', right_index=True
    )
    unitchannels = unitchannels.sort_values(
        'probe_vertical_position', ascending=False
    )

    spike_times = session.spike_times

    opto_table = session.optotagging_table

    durations = opto_table.duration.unique()
    levels = opto_table.level.unique()

    durations.sort()
    levels.sort()

    sessionID = session.metadata['ecephys_session_id']

    fig, ax = plt.subplots(len(levels), len(durations))
    fig.set_size_inches([16, 10])
    fig.suptitle(str(sessionID) + '_' + session.metadata['full_genotype'])
    for idur, duration in enumerate(durations):
        for il, level in enumerate(levels):
            opto_times = opto_table.loc[
                (opto_table['duration'] == duration)
                & (opto_table['level'] == level)
            ]['start_time'].values
            all_resp = []
            for iu, unit in unitchannels.iterrows():
                sts = spike_times[iu]

                if duration < 0.1:
                    resp = makePSTH_numba(
                        sts,
                        opto_times - 0.1,
                        0.2,
                        binSize=0.001,
                        convolution_kernel=0.002,
                    )[0]
                    resp = resp - np.mean(resp[:99])

                if duration > 0.1:
                    resp = makePSTH_numba(
                        sts,
                        opto_times - 0.25,
                        1.5,
                        binSize=0.001,
                        convolution_kernel=0.01,
                    )[0]
                    resp = resp - np.mean(resp[:249])

                all_resp.append(resp)

            all_resp = np.array(all_resp)

            im = ax[il][idur].imshow(
                all_resp, origin='lower', interpolation='none', aspect='auto'
            )
            min_clim_val = -5
            max_clim_val = 50
            im.set_clim([min_clim_val, max_clim_val])

    fig.savefig(
        os.path.join(
            nwb_validation_dir, 'optotagging', str(sessionID) + '.png'
        )
    )
    plt.close('all')


#
# pool = Pool()
# pool.map(opto_validation, nwb_paths)


#######################################
### ALL SPIKE HISTOGRAM ###############
# ash_dict = {}
# for inwb, nwb_path in enumerate(nwb_paths):


def all_spike_histogram(session):
    ash_dict = {}
    #    with NWBHDF5IO(nwb_path, 'r', load_namespaces=True) as nwb_io:
    #        session = BehaviorEcephysSession.from_nwb(nwbfile=nwb_io.read())
    #
    #    print('processing file {} of {}'.format(inwb, len(nwb_paths)))
    channels = session.get_channels()

    units = session.get_units()
    good_unit_filter = (
        (units['snr'] > 1)
        & (units['isi_violations'] < 1)
        & (units['firing_rate'] > 0.1)
    )
    units = units.loc[good_unit_filter]
    unitchannels = units.merge(
        channels, left_on='peak_channel_id', right_index=True
    )
    unitchannels = unitchannels.sort_values(
        'probe_vertical_position', ascending=False
    )

    sessionID = session.metadata['ecephys_session_id']

    spike_times = session.spike_times
    probes = unitchannels.probe_id.unique()
    ash_dict[sessionID] = {pr: [] for pr in probes}

    fig, ax = plt.subplots(len(probes), 1)
    fig.set_size_inches([14, 14])
    fig.suptitle(sessionID)
    for ip, probeid in enumerate(probes):
        probe_unitchannels = unitchannels.loc[
            unitchannels['probe_id'] == probeid
        ]

        probe_spike_times = []
        for iu, unit in probe_unitchannels.iterrows():
            sts = spike_times[iu]
            probe_spike_times.extend(sts)

        binwidth = 1
        bins = np.arange(0, np.max(probe_spike_times), binwidth)
        hist, bin_e = np.histogram(probe_spike_times, bins)
        ash_dict[sessionID][probeid] = hist

        ax[ip].plot(bin_e[1:-1], hist[1:])
        ax[ip].set_title(probeid)

    fig.savefig(
        os.path.join(
            nwb_validation_dir, 'all_spike_hists', str(sessionID) + '.png'
        )
    )
    plt.close('all')
    return ash_dict


############## VALIDATE STIMULUS RESPONSE ############
data_dict = {
    'area': [],
    'session': [],
    'active_change_response': [],
    'image_set': [],
    'omission_response': [],
    'passive_omission_response': [],
    #'shared_change_response':[],
    'passive_change_response': [],
    'rf_response': [],
    'flash_response': [],
}

ash_dict = {}
for inwb, nwb_path in enumerate(nwb_paths):
    with NWBHDF5IO(nwb_path, 'r', load_namespaces=True) as nwb_io:
        session = BehaviorEcephysSession.from_nwb(nwbfile=nwb_io.read())

    print('processing file {} of {}'.format(inwb, len(nwb_paths)))

    session_ash = all_spike_histogram(session)
    ash_dict.update(session_ash)
    opto_validation(session)

    channels = session.get_channels()

    units = session.get_units()
    good_unit_filter = (
        (units['snr'] > 1)
        & (units['isi_violations'] < 1)
        & (units['firing_rate'] > 0.1)
    )
    units = units.loc[good_unit_filter]
    unitchannels = units.merge(
        channels, left_on='peak_channel_id', right_index=True
    )

    spike_times = session.spike_times
    stimulus_presentations = session.stimulus_presentations

    change_times = stimulus_presentations.loc[
        stimulus_presentations['active']
        & stimulus_presentations['is_change']
        & (
            ~np.isin(
                stimulus_presentations['image_name'], ['im111_r', 'im083_r']
            )
        )
    ]

    change_times = change_times['start_time'].values

    passive_change_times = stimulus_presentations.loc[
        ~stimulus_presentations['active']
        & stimulus_presentations['is_change']
        & (
            ~np.isin(
                stimulus_presentations['image_name'], ['im111_r', 'im083_r']
            )
        )
    ]

    passive_change_times = passive_change_times['start_time'].values

    omission_times = stimulus_presentations.loc[
        stimulus_presentations['active'] & stimulus_presentations['omitted']
    ]['start_time'].values

    passive_omission_times = stimulus_presentations.loc[
        ~stimulus_presentations['active'] & stimulus_presentations['omitted']
    ]['start_time'].values

    #    shared_change_times = stimulus_presentations.loc[stimulus_presentations['active']&
    #                                              stimulus_presentations['is_change']&
    #                                              (np.isin(stimulus_presentations['image_name'], ['im111_r', 'im083_r']))]
    #
    #    shared_change_times = shared_change_times['start_time'].values

    flashes = stimulus_presentations[
        stimulus_presentations['stimulus_block'] == 4
    ]
    flash_times = flashes[flashes['color'] == 1].start_time.values

    gabors = stimulus_presentations[
        stimulus_presentations['stimulus_block'] == 2
    ]
    gabor_times = gabors.groupby(['position_x', 'position_y'])[
        'start_time'
    ].agg(list)

    for iu, unit in unitchannels.iterrows():

        area = unit.structure_acronym
        if 'VIS' in area:

            sts = spike_times[iu]
            cr = makePSTH_numba(
                sts,
                change_times - 1,
                2,
                binSize=0.001,
                convolution_kernel=0.005,
            )
            pcr = makePSTH_numba(
                sts,
                passive_change_times - 1,
                2,
                binSize=0.001,
                convolution_kernel=0.005,
            )
            aor = makePSTH_numba(
                sts,
                omission_times - 1,
                2,
                binSize=0.001,
                convolution_kernel=0.005,
            )
            por = makePSTH_numba(
                sts,
                passive_omission_times - 1,
                2,
                binSize=0.001,
                convolution_kernel=0.005,
            )
            flr = makePSTH_numba(
                sts,
                flash_times - 0.5,
                1,
                binSize=0.001,
                convolution_kernel=0.005,
            )
            gr = np.array(
                [
                    makePSTH_numba(
                        sts,
                        np.array(gabor_times[x][y]) - 0.5,
                        1,
                        binSize=0.001,
                        convolution_kernel=0.005,
                    )[0]
                    for x in gabors['position_x'].unique()
                    for y in gabors['position_y'].unique()
                ]
            )
            gr = gr[np.unravel_index(np.argmax(gr), gr.shape)[0]]

            #            fig,ax = plt.subplots()
            #            ax.plot(cr[0])
            #            ax.plot(pcr[0])
            #            ax.plot(aor[0])
            #            ax.plot(por[0])
            #
            #            fig, ax = plt.subplots()
            #            ax.plot(flr[0])
            #            ax.plot(gr)
            # scr = makePSTH_numba(sts, shared_change_times-1, 2, binSize=0.001, convolution_kernel=0.01

            data_dict['area'].append(area)
            data_dict['session'].append(session.metadata['ecephys_session_id'])
            data_dict['active_change_response'].append(cr[0])
            data_dict['passive_change_response'].append(pcr[0])
            data_dict['omission_response'].append(aor[0])
            data_dict['passive_omission_response'].append(por[0])
            data_dict['flash_response'].append(flr[0])
            data_dict['rf_response'].append(gr)
            # data_dict['shared_change_response'].append(scr[0])
            data_dict['image_set'].append(
                session.task_parameters['session_type']
            )


H_NOT_NOVEL = [
    1099598937,
    1099869737,
    1104052767,
    1104297538,
    1109680280,
    1109889304,
    1108335514,
    1108528422,
    1116941914,
    1117148442,
    1118324999,
    1118512505,
    1119946360,
    1120251466,
    1130113579,
    1130349290,
    1132595046,
    1132752601,
    1127072792,
    1127256514,
    1131502356,
    1131648544,
    1128520325,
    1128719842,
    1139846596,
    1140102579,
]

# fig_save_dir = r"C:\Users\svc_ccg\Desktop\Presentations\SAC 2022"
# areas_of_interest = ['VISp', 'VISl', 'VISal', 'VISrl',
#                     'VISam', 'VISpm', 'LP', 'LGd', 'TH',
#                     'MRN', 'CA3', 'SCig', 'SCiw']
#
data_df = pd.DataFrame(data_dict)
#
# for area in areas_of_interest:
#    fig, ax = plt.subplots()
#    fig.suptitle(area)
#
#    novel = []
#    familiar = []
#    ainds = np.where(np.array(data_dict['area'])==area)[0]
#    print(area, len(ainds))
#    for ai in ainds:
#        image_set = data_dict['image_set'][ai]
#        s_id = data_dict['session'][ai]
#        cr = data_dict['change_response'][ai]
#        if '_G' in image_set:
#            if s_id in H_NOT_NOVEL:
#                novel.append(cr)
#            else:
#                familiar.append(cr)
#        else:
#            if s_id in H_NOT_NOVEL:
#                familiar.append(cr)
#            else:
#                novel.append(cr)
#
#    for color, exp in zip(['r', 'b'], [novel, familiar]):
#        mean = np.mean(exp, axis=0)
#        sem = np.std(exp, axis=0)/(len(exp)**0.5)
#
#        x = np.linspace(-1, 1, len(cr))
#        ax.plot(x, mean, color)
#        ax.fill_between(x, mean+sem, mean-sem, color=color, alpha=0.25)
#
#    #ax.set_xlim([-0.2, 0.5])
#    fig.savefig(os.path.join(fig_save_dir, area+'_change_response.pdf'))
#
#
# for area in areas_of_interest:
#    fig, ax = plt.subplots()
#    fig.suptitle(area)
#
#    novel_unshared = []
#    novel_shared = []
#    fam_unshared = []
#    fam_shared = []
#    ainds = np.where(np.array(data_dict['area'])==area)[0]
#    print(area, len(ainds))
#    for ai in ainds:
#        image_set = data_dict['image_set'][ai]
#        s_id = data_dict['session'][ai]
#        cr = data_dict['change_response'][ai]
#        scr = data_dict['shared_change_response'][ai]
#        if '_G' in image_set:
#            if s_id in H_NOT_NOVEL:
#                novel_unshared.append(cr)
#                novel_shared.append(scr)
#            else:
#                fam_unshared.append(cr)
#                fam_shared.append(scr)
#        else:
#            if s_id in H_NOT_NOVEL:
#                fam_unshared.append(cr)
#                fam_shared.append(scr)
#            else:
#                novel_unshared.append(cr)
#                novel_shared.append(scr)
#
#    for color, exp in zip(['r', 'b', 'orange', 'teal'],
#                          [novel_unshared, fam_unshared, novel_shared, fam_shared]):
#        mean = np.mean(exp, axis=0)
#        sem = np.std(exp, axis=0)/(len(exp)**0.5)
#
#        x = np.linspace(-1, 1, len(cr))
#        ax.plot(x, mean, color)
#        ax.fill_between(x, mean+sem, mean-sem, color=color, alpha=0.25)
#
#    #ax.set_xlim([-0.2, 0.5])
#    fig.savefig(os.path.join(fig_save_dir, area+'_change_response_shared_vs_unshared.pdf'))


for sessionID in data_df.session.unique():

    sdf = data_df.loc[data_df['session'] == sessionID]

    for visarea in sdf.area.unique():

        sadf = sdf.loc[sdf['area'] == visarea]
        fig, ax = plt.subplots()
        fig.set_size_inches([10, 6])

        fig_inset, ax_inset = plt.subplots()

        fig.suptitle(visarea)
        figrf, axrf = plt.subplots()
        figrf.suptitle(visarea)
        figrf.set_size_inches([10, 6])

        for plot in [
            'active_change_response',
            'passive_change_response',
            'omission_response',
            'passive_omission_response',
        ]:

            mean = np.mean(sadf[plot], axis=0)
            ax.plot(mean)
            ax_inset.plot(np.arange(-100, 200), mean[900:1200])

        for plot in ['rf_response', 'flash_response']:

            mean = np.mean(sadf[plot], axis=0)
            axrf.plot(np.arange(-100, 200), mean[400:700])

        ax.legend(
            [
                'active_change_response',
                'passive_change_response',
                'omission_response',
                'passive_omission_response',
            ]
        )
        fig.savefig(
            os.path.join(
                nwb_validation_dir,
                'image_vis_response',
                str(sessionID) + '_' + visarea + '.png',
            )
        )

        ax_inset.legend(
            [
                'active_change_response',
                'passive_change_response',
                'omission_response',
                'passive_omission_response',
            ]
        )
        fig_inset.savefig(
            os.path.join(
                nwb_validation_dir,
                'image_vis_response',
                str(sessionID) + '_' + visarea + '_latency.png',
            )
        )

        axrf.legend(['rf_response', 'flash_response'])
        figrf.savefig(
            os.path.join(
                nwb_validation_dir,
                'rf_vis_response',
                str(sessionID) + '_' + visarea + '.png',
            )
        )

        plt.close('all')


# ### SEIZURE DETECTION ###

# df = pd.read_excel(r"C:\Users\svc_ccg\ccb_onedrive\OneDrive - Allen Institute\VBN_release_shared.xlsx")
# manual_seizure = df.loc[[s in [True, '?', 'minor', 'minor?', 'TRUE?'] for s in df.Seizure]]['full_id']
# manual_seizure_sessions = [int(s[:10]) for s in manual_seizure.values]
# opt_sessions = df.loc[[s not in [False, np.nan] for s in df.Seizure]]['full_id']

# seizure_dict = {}
# zscored = {}
# for sid in ash_dict:
#     print(sid)
#     ss = ash_dict[sid]
#     seizure_dict[sid] = []
#     zscored[sid] = {}
#     for probeid in ss:

#         p = ss[probeid]

#         detrended = p - scipy.signal.medfilt(p, 1001)
#         detrended_zscore = (detrended - np.mean(detrended))/np.std(detrended)
#         zscored[sid][probeid] = detrended_zscore

#         candidates = np.where(detrended > np.mean(detrended) + 5*np.std(detrended)) #was 5
#         if len(candidates)>0:
#             candidates = candidates[0]
#             seizures = []
#             for c in candidates:
#                 baseline = detrended[c-110:c-10]
#                 after = detrended[c+5:c+50]
#                 if np.mean(after) < np.mean(baseline) - 2*np.std(baseline):
#                     if not 8800<c< 8850: #Don't take seizures that are right at the beginning of opto TODO: find opto start times for each session
#                         seizures.append(c)
#                 #seizures.append(c)
#         seizure_dict[sid].append(seizures)

# interictal = 60
# seizure_time_dict = {}
# for s in seizure_dict:
#     ss = seizure_dict[s]
#     st = []
#     seizure_times = np.sort(np.array([item for sublist in ss for item in sublist]))
#     if len(seizure_times) > 0:
#         st = seizure_times[np.insert(np.diff(seizure_times)>interictal, 0, True)]

#     seizure_time_dict[s] = st


# flatten = lambda x: [item for sub in x for item in sub]
# spike_dip_seizure_sessions = [s for s in seizure_dict if len(flatten(seizure_dict[s]))>0]
# spike_seizure_sessions = [s for s in zscored if np.max(flatten([zscored[s][pid] for pid in zscored[s]]))>10]
# diff = np.setdiff1d(spike_seizure_sessions, spike_dip_seizure_sessions)

# for criterion, savedir in zip([spike_dip_seizure_sessions, diff], ['std_5_spike_std_2_dip', 'std_10_spike']):
#     for s in criterion:
#         ss = seizure_dict[s]

#         #seizure_times = [item for sublist in ss for item in sublist]
#         seizure_times = seizure_time_dict[s]
#         fig, ax = plt.subplots(6)
#         fig.suptitle(s)
#         for ip, p in enumerate(ash_dict[s]):
#             if 'dip' in savedir:
#                 for st in seizure_times:
#                     ax[ip].axvline(st, c='r', alpha=0.5)
#             else:
#                 sts = np.where(zscored[s][p]>10)[0]
#                 for st in sts:
#                     ax[ip].axvline(st, c='r', alpha=0.5)
#             ax[ip].plot(ash_dict[s][p])

#         fig.set_size_inches([14,8])
#         fig.savefig(os.path.join(nwb_validation_dir, 'seizures', savedir, str(s)+'.png'))

#     plt.close('all')

# weird_visual_sessions = [1062755779, 1067790400, 1077897245, 1079275221, 1084428217, 1084939136]
# for s in spike_dip_seizure_sessions:
# #for s in manual_in_release:
#     #if s not in spike_dip_seizure_sessions:
#         ss = seizure_dict[s]

#         seizure_times = [item for sublist in ss for item in sublist]
#         fig, ax = plt.subplots(6)
#         fig.suptitle(s)
#         for ip, p in enumerate(ash_dict[s]):
#             if 'dip' in savedir:
#                 for st in seizure_times:
#                     ax[ip].axvline(st, c='r', alpha=0.5)
#             else:
#                 sts = np.where(zscored[s][p]>10)[0]
#                 for st in sts:
#                     ax[ip].axvline(st, c='r', alpha=0.5)
#             ax[ip].plot(ash_dict[s][p])

#         fig.set_size_inches([14,8])
#         fig.savefig(os.path.join(nwb_validation_dir, 'seizures', 'manual_annotation', str(s)+'.png'))


# #for s in seizure_dict:
# for s in diff:
#     ss = seizure_dict[s]
#     seizure_times = [item for sublist in ss for item in sublist]
#     num_seizures = len(seizure_times)
#     if num_seizures>0:
#         print(s)
#         fig, ax = plt.subplots(6)
#         fig.suptitle(s)
#         for ip, p in enumerate(ash_dict[s]):
#             for st in seizure_times:
#                 ax[ip].axvline(st, c='r', alpha=0.5)
#             ax[ip].plot(ash_dict[s][p])


# test_zs = np.arange(1, 20, 0.5)
# seizure_count = np.zeros(len(test_zs))
# rand_count = np.zeros(len(test_zs))

# rand = [[np.random.poisson(1650, 9000) for i in range(6)] for j in range(149)]
# rand_zscored = [[(rr-np.mean(rr))/np.std(rr) for rr in r] for r in rand]

# for iz, z in enumerate(test_zs):

#     for ind, s in enumerate(zscored):
#         seiz = 0
#         rseiz = 0
#         zs = zscored[s]
#         rzs = rand_zscored[ind]
#         for pind, p in enumerate(zs):
#             if any(zs[p]>z):
#                 seiz = 1
#             if any(rzs[pind]>z):
#                 rseiz = 1

#         seizure_count[iz] += seiz
#         rand_count[iz] += rseiz

# plt.plot(test_zs,seizure_count)
# plt.plot(test_zs,rand_count)

# for ind, s in enumerate(zscored):
#     seiz = 0
#     zs = zscored[s]
#     for pind, p in enumerate(zs):
#         if any(zs[p]>15):
#             seiz = 1

#     if seiz==1:
#         fig, ax = plt.subplots()
#         [ax.plot(zs[p]) for p in zs]

# df['Abnormal Activity']= None

# for s in spike_dip_seizure_sessions:
#     df.loc[df.full_id.str.contains(str(s)),'Abnormal Activity']=[list(seizure_time_dict[s])]

# writer = pd.ExcelWriter(r"C:\Users\svc_ccg\ccb_onedrive\OneDrive - Allen Institute\VBN_seizure_flag.xlsx", engine='xlsxwriter')
# df.to_excel(writer, sheet_name='Sheet1', index=False)
# writer.save()

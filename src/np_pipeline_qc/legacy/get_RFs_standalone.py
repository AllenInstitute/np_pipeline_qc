# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 17:25:46 2020

@author: svc_ccg
"""
import numpy as np
import os
#from visual_behavior.ophys.sync import sync_dataset
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import analysis
import probeSync_qc as probeSync
import data_getters
import logging
import sys
from analysis import save_figure
import pickle
# sys.path.append("..")
from sync_dataset import Dataset as sync_dataset
import scipy.signal
import argparse


def get_RFs(probe_dict, mapping_data, first_frame_offset, FRAME_APPEAR_TIMES, 
            FIG_SAVE_DIR, ctx_units_percentile = 40, return_rfs=False, 
            response_thresh=20, filter_on_significant=True, tile_rfs=True, chan_bin=9, max_rows=20, max_cols=20, prefix='',
            save_rf_mat=False, plot=True, stimulus_index=0): 
    
    ### PLOT POPULATION RF FOR EACH PROBE ###
    print('Return rfs set to {}'.format(return_rfs))
    rfs = {p:{k:[] for k in ['peak_channel', 'unitID', 'rfmat']} for p in probe_dict}
    for p in probe_dict:
        try:
            print(f'########## Getting RFs for probe {p} ###########')
            u_df = probe_dict[p]
            #good_units = u_df[(u_df['quality']=='good')&(u_df['snr']>1)]
            good_units = u_df[(u_df['snr']>1)&(u_df['isi_viol']<1)&(u_df['firing_rate']>0.1)]

            ctx_bottom_chan = np.percentile(good_units['peak_channel'], 100-ctx_units_percentile)
            #spikes = good_units.loc[good_units['peak_channel']>ctx_bottom_chan]
            spikes = good_units
            ctx_rmats = []
            for ind, s in spikes.iterrows():
                rmat = analysis.plot_rf(mapping_data, s['times'].flatten(), first_frame_offset, FRAME_APPEAR_TIMES, stimulus_index=stimulus_index)
                if filter_on_significant:
                    significant = get_significant_rf(rmat)
                else:
                    significant = rmat.max()>response_thresh
                if significant:
                    rfs[p]['peak_channel'].append(s['peak_channel'])
                    rfs[p]['unitID'].append(s['Unnamed: 0'])
                    rfs[p]['rfmat'].append(rmat)
                    if s['peak_channel']>ctx_bottom_chan:
                        ctx_rmats.append(rmat/rmat.max())
                
            rmats_normed_mean = np.nanmean(ctx_rmats, axis=0)
         
            if plot:
                rfig = plt.figure(constrained_layout=True, figsize=[6,6])
                title = p + ' population RF: {} units'.format(len(ctx_rmats))   
                rfig.suptitle(title, color='w')
                
                nrows, ncols = 10,10
                gs = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=rfig)
                
                ax1 = rfig.add_subplot(gs[0:nrows-1, 0:ncols-1])
                ax2 = rfig.add_subplot(gs[0:nrows-1, ncols-1])
                ax3 = rfig.add_subplot(gs[nrows-1, 0:ncols-1])
                
                ax1.imshow(np.mean(rmats_normed_mean, axis=2), origin='lower')
                ax1.set_xticks([], minor=[])
                ax1.set_yticks([], minor=[])
                
                ax3.imshow(np.vstack((np.arange(-45, 46), np.arange(-45, 46))), cmap='jet', clim=[-60, 60])
                ax3.set_xticks([0, 45, 90])
                ax3.set_xticklabels([-45, 0, 45])
                ax3.set_yticks([], minor=[])
                ax3.set_xlabel('Azimuth')
                
                ax2.imshow(np.hstack((np.arange(-45, 46)[:,None], np.arange(-45, 46)[:,None])), cmap='jet_r', clim=[-60, 60])
                ax2.yaxis.tick_right()
                ax2.set_yticks([0, 45, 90])
                ax2.set_yticklabels([-45, 0, 45])
                ax2.set_xticks([], minor=[])
                ax2.yaxis.set_label_position("right")
                ax2.set_ylabel('Elevation', rotation=270)
                
                save_path = os.path.join(FIG_SAVE_DIR, prefix + p + ' population RF.png')
                print(save_path)
                save_figure(rfig, save_path)
                #rfig.savefig(os.path.join(FIG_SAVE_DIR, title + '.png'))
            
        
        except Exception as E:
            logging.error(f'{p} failed: {E}')
            print(E)
        
        
    if tile_rfs:
        print('tiling')
        try:
            plot_tiled_rfs(rfs, FIG_SAVE_DIR, chan_bin, max_rows, max_cols, prefix)
        except Exception as E:
            logging.error('Failed to tile rfs: {}'.format(E))
            
    if save_rf_mat:
        rf_save_dir = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\QC\rf_summary"
        with open(os.path.join(rf_save_dir, prefix +'rfmats.npy'), 'wb') as fp:
            pickle.dump(rf_mat, fp)
                
    if return_rfs:
        return rfs
    
    
def plot_tiled_rfs(rfdict, FIG_SAVE_DIR, chan_bin = 8, max_rows=20, max_cols=20, prefix=''):
    
    for p in rfdict:
        
        peakchans = rfdict[p]['peak_channel']
        rfs = np.array(rfdict[p]['rfmat'])
        
        lowest_chan = np.min(peakchans)
        highest_chan = np.max(peakchans)
        
        num_bins = (highest_chan - lowest_chan)/chan_bin
        if num_bins>max_cols:
            chan_bin = int(np.ceil((highest_chan-lowest_chan)/max_cols))
            
        chan_bins = np.arange(lowest_chan, highest_chan+chan_bin, chan_bin)
        rf_bins = np.digitize(peakchans, chan_bins)-1
        
        highest_bin_count = np.max([np.sum(rf_bins==b) for b in np.unique(rf_bins)])
        nrows = np.min([len(chan_bins), max_rows])
        ncols = np.min([highest_bin_count, max_rows])
        
        fig, axes = plt.subplots(nrows, ncols, figsize=[ncols, nrows])
        title = p + '_RFs_by_depth'   
        fig.suptitle(title, color='w')
        for ax in axes.flat:
            ax.set_visible(False)
        
        for ib, b in enumerate(np.unique(rf_bins)):
            b_rfs = rfs[rf_bins==b]
            if len(b_rfs)>ncols:
                b_rfs = b_rfs[:ncols]
            for ir, rr in enumerate(b_rfs):
                rrmean = np.mean(rr, axis=2)
                ax = axes[nrows - 1 - b][ir]
                ax.set_visible(True)
                ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
                ax.imshow(rrmean, origin='lower')
                if ir==0:
                    ax.tick_params(labelleft=True)
                    ax.set_ylabel(chan_bins[b], rotation=0)
                    ax.set_yticklabels([])

        save_figure(fig, os.path.join(FIG_SAVE_DIR, prefix + title + '.png'))

    
def get_significant_rf(rfmat, nreps=1000, conv=2):
    
    if rfmat.ndim>2:
        rfmat = np.mean(rfmat, axis=2)
    
    conv_mat = np.ones((2,2))
    rf_conv = scipy.signal.convolve2d(rfmat, conv_mat, 'same')/4
    
    shuffled = []
    rf_shuff = np.copy(rfmat)
    for rep in np.arange(nreps):
        flat = rf_shuff.flatten()
        np.random.shuffle(flat)
        unflat = flat.reshape(rfmat.shape)
        unflat_conv = scipy.signal.convolve2d(unflat, conv_mat, 'same')/4
        shuffled.append(unflat_conv)
    
    shuff_max = [s.max() for s in shuffled]
    percentile_95 = np.percentile(shuff_max, 95)
    
    return rf_conv.max() > percentile_95
                  

if __name__ == "__main__":
    
    # run as standalone script
    print('trying to argparse')
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save_dir", default='')
    
    args = parser.parse_args()
    
    experiment_id = args.experiment_id
    save_rf_npy = args.save
    save_dir = args.save_dir
    
    if save_rf_npy:
        print('rf mat will save to {}'.format(save_dir))
    
    print(experiment_id)
    
    d = data_getters.local_data_getter(base_dir=experiment_id)
    paths = d.data_dict
    
    FIG_SAVE_DIR = os.path.join(r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\QC", 
                                paths['es_id']+'_'+paths['external_specimen_name']+'_'+paths['datestring'])
    
    figure_prefix = paths['external_specimen_name'] + '_' + paths['datestring']

    if not os.path.exists(FIG_SAVE_DIR):
        os.mkdir(FIG_SAVE_DIR)
    
    print('Saving plots to {}'.format(FIG_SAVE_DIR))

    ### GET FILE PATHS TO SYNC AND PKL FILES ###
    SYNC_FILE = paths['sync_file']
    # BEHAVIOR_PKL = paths['behavior_pkl']
    # REPLAY_PKL = paths['replay_pkl']
    MAPPING_PKL = paths['mapping_pkl']

    try:
        syncDataset = sync_dataset(SYNC_FILE)
    except Exception as e:
        logging.error('Error reading sync file: {}'.format(e))

    try:
        mapping_data = pd.read_pickle(MAPPING_PKL)
    except Exception as e:
        logging.error('Error reading mapping pkl file: {}'.format(e))

    # replay_data = pd.read_pickle(REPLAY_PKL)
    
    
    ### PLOT FRAME INTERVALS ###
    vr, vf = probeSync.get_sync_line_data(syncDataset, channel=2)
    
    # behavior_frame_count = behavior_data['items']['behavior']['intervalsms'].size + 1
    mapping_frame_count = mapping_data['intervalsms'].size + 1
    # replay_frame_count = replay_data['intervalsms'].size + 1
    
    MONITOR_LAG = 0.036
    FRAME_APPEAR_TIMES = vf + MONITOR_LAG  
    
    ### CHECK THAT NO FRAMES WERE DROPPED FROM SYNC ###
    # total_pkl_frames = (behavior_frame_count +
    #                     mapping_frame_count +
    #                     replay_frame_count) 

    # print('frames in pkl files: {}'.format(total_pkl_frames))
    # print('frames in sync file: {}'.format(len(vf)))
    
    #infer start frames for stimuli
    start_frame = probeSync.get_frame_offsets(syncDataset, [mapping_frame_count])
    
    if start_frame is not None:
        print('RF mapping started at frame {}, or experiment time {} seconds'.format(start_frame[0], start_frame[0]/60.))
        
        probe_dict = probeSync.build_unit_table(paths['data_probes'], paths, syncDataset)
        probe_dict_old_format = {}
        for p in probe_dict['probe'].unique():
            probe_dict_old_format[p] = probe_dict.loc[probe_dict['probe']==p]
            
        rf_mat = get_RFs(probe_dict_old_format, mapping_data, start_frame[0], FRAME_APPEAR_TIMES, FIG_SAVE_DIR, 
                         return_rfs=True, prefix=figure_prefix, plot=False, tile_rfs=False)
        rf_save_dir = save_dir
        if save_rf_npy:
            with open(os.path.join(rf_save_dir, paths['es_id']+'_'+paths['external_specimen_name']+'_'+paths['datestring']+'rfmats.npy'), 'wb') as fp:
                pickle.dump(rf_mat, fp)
#            np.save(os.path.join(rf_save_dir, paths['es_id']+'_'+paths['external_specimen_name']+'_'+paths['datestring']+'rfmats.npy'), rf_mat)
    
    else:
        logging.error('Could not find mapping stim start frame')


    

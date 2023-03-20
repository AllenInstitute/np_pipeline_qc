# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:39:59 2021

@author: svc_ccg
"""

import pandas as pd
import os
import EcephysBehaviorSession as ebs
from matplotlib import pyplot as plt
import numpy as np
from numba import njit


def get_ctx_inds(unit_table):
    probe_grouped = unit_table.groupby('probe')
    ctx_inds = []
    for probe, pgroup in probe_grouped:
        
        top_channel = pgroup.loc[pgroup['quality']=='good']['peak_channel'].max()
        bottom_ctx = top_channel - 70
        pctx = pgroup.loc[pgroup['peak_channel']>bottom_ctx]
        ctx_inds.extend(pctx.index.values)
    return ctx_inds


def get_ctx_df(unit_table):
    
    ctx_inds = get_ctx_inds(unit_table)
    return unit_table.loc[ctx_inds]


@njit     
def makePSTH_numba_pertrial(spikes, startTimes, windowDur, binSize=0.001, convolution_kernel=0.05):
    spikes = spikes.flatten()
    startTimes = startTimes - convolution_kernel/2
    windowDur = windowDur + convolution_kernel
    bins = np.arange(0,windowDur+binSize,binSize)
    convkernel = np.ones(int(convolution_kernel/binSize))
    counts = np.zeros((len(startTimes), bins.size-1))
    for i,start in enumerate(startTimes):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts[i] = np.histogram(spikes[startInd:endInd]-start, bins)[0]
    
 
    out = np.zeros((counts.shape[0], len(bins[:-convkernel.size-1])))
    for ic in range(counts.shape[0]):
        c = counts[ic]
        c = np.convolve(c, convkernel)/(binSize*convkernel.size)
        out[ic] = c[convkernel.size-1:-convkernel.size]
    
    return out, bins[:-convkernel.size-1]



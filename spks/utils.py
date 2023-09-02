import numpy as np
import os
import sys
from os.path import join as pjoin
from natsort import natsorted
from glob import glob
import pandas as pd
import torch

def tensor_to_numpy(X):
    '''Converts a tensor to numpy array.''' 
    return X.to('cpu').numpy()

def free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def chunk_indices(data, axis = 0, chunksize = 60000, min_chunksize = 512):
    '''
    Gets chunk indices for iterating over the dataset in evenly sized chunks
    
    indices = chunk_indices(data)

    Joao Couto - May 2020
    '''
    chunks = np.arange(0,data.shape[axis], chunksize, dtype = int)
    if (data.shape[1] - chunks[-1]) < min_chunksize:
        chunks[-1] = data.shape[axis]
    if not chunks[-1] == data.shape[axis]:
        chunks = np.hstack([chunks, self.shape[1]])
    return [[chunks[i], chunks[i+1]] for i in range(len(chunks)-1)]


from numpy.lib.stride_tricks import as_strided

def _check_arg(x, xname):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('%s must be one-dimensional.' % xname)
    return x

def xcorr(x, y, maxlag):
    """
    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return vaue has length 2*maxlag + 1.

    Adapted from code from Warren Weckesser (stackoverflow).
    
    Joao Couto - January 2016
    """
    x = _check_arg(x, 'x')
    y = _check_arg(y, 'y')
    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    return T.dot(px)

def convolve_with_kernel(fr,sigma = 5,dt = 1,kernel = 'exp'):
    ''' 
    Convolves with a kernel.
        - kernel can be exp or gauss if it is not a string you should provive a 1d vector smaller than fr (the kernel to convolve with).
    ''' 
    dt *= 1e-3
    sigma *= 1e-3
    xx = np.arange(-3*sigma,3*sigma,dt)
    if not type(kernel) is str:
        s = kernel
    elif kernel == 'gauss':
        s =  np.exp(-(xx)**2/(2*sigma)**2)/(np.sqrt(2.*np.pi)*sigma)
        s = s/np.sum(s)
    elif kernel == 'exp':
        s =  np.exp(-1. * np.sqrt(2)*np.abs(xx/sigma))/(np.sqrt(2.)*sigma)
        s = s/np.sum(s)
    return np.convolve(fr,s,'same')

def find_spikes(dat,thresh = None,wpre=16,wpos=20,threshstd=6):
    ''' Find spikes and extract sample times and waveforms '''
    tmp = np.zeros(shape=dat.shape)
    if thresh is None:
        thresh = compute_spike_threshold(dat,threshstd)
    tmp[dat<-thresh] = 1
    tstamps = np.where(np.diff(tmp)>0)[0]
    # align...
    for i,t in enumerate(tstamps):
        tmp = dat[t-wpre:t+wpos]
        tmpmax = np.argmin(tmp)
        tstamps[i] = t-wpre+tmpmax
        #extract waveforms
        waves = np.zeros(shape=(len(tstamps),wpre+wpos))
        for i,t in enumerate(tstamps):
            try:
                waves[i,:] = dat[t-wpre:t+wpos]
            except:
                print('Failed for spike {0}'.format(t))
    return tstamps,waves

def compute_spike_threshold(x,stdmin=4):
    ''' Compute spike threshold from filtered raw trace.
    Uses the formula from R. Quiroga, Z. Nadasdy, and Y. Ben-Shaul:
       thr = stdmin*sigma_n ,with
       sigma_n = median(|x|/0.6745)
 NOTE: 
   Default stdmin is 4.
   In the WaveClus batch scripts stdmin is set to 5.
    Joao Couto - January 2016    
    '''
    return stdmin * np.median(np.abs(x))/0.6745;

def whitening_matrix(x, fudge=1e-18):
    """
    wmat = whitening_matrix(dat, fudge=1e-18)
    Compute the whitening matrix.
        - dat is a matrix nsamples x nchannels
    Apply using np.dot(dat,wmat)
    Adapted from phy
    """
    assert x.ndim == 2
    ns, nc = x.shape
    x_cov = np.cov(x, rowvar=0)
    assert x_cov.shape == (nc, nc)
    d, v = np.linalg.eigh(x_cov)
    d = np.diag(1. / np.sqrt(d + fudge))
    w = np.dot(np.dot(v, d), v.T)
    return w

def gaussian_function(sigma):
    x = np.arange(np.floor(-3*sigma),np.ceil(3*sigma)) # will always be even number of indices
    kernel = np.exp(-(x/sigma)**2/2)/(sigma*np.sqrt(2*np.pi))
    return kernel

def alpha_function(N, amplitude = 1, t_rise = 2, t_decay = 250, srate = 1000.,norm = True):

    t_rise = t_rise/srate;
    t_decay = t_decay/srate;
    
    fun_max  = (t_rise*t_decay/(t_decay-t_rise)) * np.log(t_decay-t_rise);
    normalization_factor = 1; #%(exp(-fun_max/t_rise) - exp(-fun_max/t_decay))/(t_rise-t_decay);
    ii = np.arange(0,N)
    kernel = np.hstack([np.zeros(N),
                        amplitude*(1.0/(normalization_factor*(t_decay-t_rise))) * (np.exp(-((ii/srate)/t_decay))
                                                                                   - np.exp(-(ii/srate)/t_rise))])
    if norm:
        kernel /= np.sum(kernel)
    return kernel

def binary_spikes(spks,edges,kernel = None):
    ''' Create a vector of binary spikes. Optionally convolve with a kernel
    binsize = 0.001
    edges = np.arange(0,5,binsize) this should be in seconds
    bspks = binary_spikes(spks,edges)/binsize
    Joao Couto - March 2016
    Modified by Max Melin
    '''
    if kernel is not None:
        binwidth_s = np.mean(np.diff(edges))
        n_pad = kernel.size / 2

        start_pad = np.arange(-binwidth_s * n_pad, 0, binwidth_s) + edges[0]
        end_pad = np.arange(0, binwidth_s * n_pad, binwidth_s) + edges[-1]

        padded_edges = np.concatenate((start_pad[:-1], edges, end_pad[1:]))

        bins = [np.histogram(sp,padded_edges)[0] for sp in spks]
        bins = [np.convolve(a,kernel,'valid') for a in bins] #'valid' avoids edge artifacts
    else:
        bins = [np.histogram(sp,edges)[0] for sp in spks]
    return np.vstack(bins)

def align_raster_to_event(event_times, spike_times, pre_seconds, post_seconds):
    """create aligned rasters relative to event_times

    Parameters
    ----------
    event_times : list or ndarray
        a list or numpy array of event times to be aligned to
    spike_times : list or ndarray
        a list spike times for one cluster
    pre_seconds : float
        grab _ seconds before event_times for alignment, by default 1
    post_seconds : float
        grab _ seconds after event_times for alignment, by default 2

    Returns
    -------
    list
        a list of aligned rasters for each event_times
    """    
    #TODO: add option to pass a list maximum pre and post times, so we can truncate data that bleeds into other events. Useful for multiple event alignment. 
    event_rasters = []
    for i, event_time in enumerate(event_times):
        relative_spiketimes = spike_times - event_time
        spks = relative_spiketimes[np.logical_and(relative_spiketimes <= post_seconds, relative_spiketimes >= -pre_seconds)]
        event_rasters.append(np.array(spks))
    return event_rasters

def compute_firing_rate(event_times, spike_times, pre_seconds, post_seconds, binwidth_ms=25, kernel=None):
    '''compute the PETH for one neuron'''
    binwidth_s = binwidth_ms/1000
    event_times = discard_nans(event_times) 
    
    rasters = align_raster_to_event(event_times, 
                                spike_times,
                                pre_seconds,
                                post_seconds)

    #construct timebins separately for pre and post so that the alignment event occurs at the center of a timebin
    pre_event_timebins = np.arange(-pre_seconds, 0, binwidth_s)
    post_event_timebins = np.arange(0, post_seconds+binwidth_s, binwidth_s)
    timebin_edges = np.append(pre_event_timebins, post_event_timebins)

    event_index = pre_event_timebins.size

    psth_matrix = binary_spikes(rasters, timebin_edges, kernel=kernel) / binwidth_s # divide by binwidth to get a rate rather than count
    return psth_matrix, event_index

from scipy.interpolate import interp2d
from scipy.signal import ellip, filtfilt,butter

def bandpass_filter(X,srate,band=[3,300]):
    b, a = ellip(4, 0.1, 40, np.array(band)/(srate/2.),btype='bandpass')
    return filtfilt(b, a, X,axis = 0)#, method="gust"

def current_source_density(lfp,chmap, chanspacing=60, interpolate=False):
    # Interpolate so that we get even sampling
    selchannels = np.array(chmap.ichan)
    ux = np.unique(chmap.x)
    ix = np.argmax([np.sum(chmap.x==u) for u in ux])
    chidx = chmap.x==ux[ix]
    y = np.array(chmap[chidx].y)
    duration = lfp.shape[1]
    x = np.arange(duration)
    z = lfp[chmap[chidx].ichan,:]
    f = interp2d(x,y,z)
    ny = np.arange(np.min(y)-chanspacing,np.max(y)+chanspacing,chanspacing)
    nlfp = f(x,ny)
    # duplicate outmost channels
    csd = np.empty((nlfp.shape[0]-2,nlfp.shape[1]))
    smoothed_lfp = np.empty_like(nlfp)
    for i in range(csd.shape[0]):
        smoothed_lfp[i+1,:] = (1./4.) *(nlfp[i,:] + 2.*nlfp[i+1,:] + nlfp[i+2,:])
    smoothed_lfp[0,:] = (1./4.) *(3.*nlfp[0,:] + nlfp[1,:])
    smoothed_lfp[-1,:] = (1./4.) *(3.*nlfp[-1,:] + nlfp[-2,:])
    smoothed_lfp = smoothed_lfp
    for i in range(csd.shape[0]):
        csd[i,:] = -(1./(chanspacing*1.e-3)**2.)*(smoothed_lfp[i,:]-2.*smoothed_lfp[i+1,:]+smoothed_lfp[i+2,:])
    f = interp2d(x,np.linspace(np.min(y)-chanspacing,np.max(y)+chanspacing,csd.shape[0]),csd)
    ny = np.arange(np.min(y)-chanspacing,np.max(y)+chanspacing,5.)
    return f(x,ny),smoothed_lfp[:,1:-1]

def discard_nans(input_array):
    """discards nan's from an array and throws a text warning of how many NaN's got discarded
    """
    if np.sum(np.isnan(input_array)) > 0:
        print(f'Discarding {np.sum(np.isnan(input_array))} NaN''s from event_times')  #TODO: print out the specific variable name
        return input_array[~np.isnan(input_array)]
    else:
        return input_array



import numpy as np
import os
import sys
from os.path import join as pjoin
from natsort import natsorted
from glob import glob
import pandas as pd
import torch
from numpy.lib.stride_tricks import as_strided
from functools import partial
from multiprocessing import Pool, cpu_count
from scipy.stats import median_abs_deviation 
from pathlib import Path
import re
from tqdm import tqdm
from multiprocessing.pool import Pool,ThreadPool
from multiprocessing import cpu_count
import h5py as h5
import datetime
import shutil
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

mad = lambda x : median_abs_deviation(x,scale='normal',nan_policy='omit')

class TemporaryArrayOnDisk(np.memmap):
    """TemporaryArrayOnDisk extends np.memmap in two ways
        - provides a destructor function that will delete the memmap file when all variable references to it are gone.
        - will accept an absolute binary path, or will randomly generate a filename if given a target directory

    Max Melin - 2023
    """
    def __new__(cls, fast_drive_path,*args, **kwargs):
        FILE_EXTENSION = '.bin'
        if os.path.isdir(fast_drive_path):
            import random
            import string
            FILE_STRING_LENGTH = 20 # the length of the randomly generated filename
            random_file_name = ''.join(random.choices(string.ascii_letters, k=FILE_STRING_LENGTH)) + FILE_EXTENSION
            filename = os.path.join(fast_drive_path, random_file_name)
        else:
            _, file_extension = os.path.splitext(fast_drive_path)
            assert file_extension == FILE_EXTENSION
            filename = fast_drive_path
        self = super().__new__(cls, filename, *args, **kwargs)
        return self

    def __del__(self):
        # The destructor will delete the file when the object is deleted or goes out of scope
        # but it will not handle a system exit while the kernel is running.
        if self._mmap:
            self._mmap.close()
            os.remove(self.filename)

def create_temporary_folder(path, prefix='spks'):
    date = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    rand = ''.join(np.random.choice([a for a in 'qwertyuiopasdfghjklzxcvbnm1234567890'],size=4))
    foldername = pjoin(path,f'{prefix}_{date}_{rand}')
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    return foldername 

def tensor_to_numpy(X):
    '''Converts a tensor to numpy array.''' 
    return X.to('cpu').numpy()

def numpy_to_tensor(X, device = 'cpu'):
    '''Converts a numpy array to tensor to numpy array.'''
    if X.dtype in [np.uint32]:
        dtype = np.int64
    else:
        dtype = X.dtype
    return torch.from_numpy(X.astype(dtype)).to(device)

def check_cuda(device):
    if device is None:
        device = 'cuda'
    if device == 'cuda':
        if not torch.cuda.is_available():
            print('Torch does not have access to the GPU; setting device to "cpu"')
            device = 'cpu'
    return device


def free_gpu():
    ''' free torch gpu memory '''
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
        chunks = np.hstack([chunks, data.shape[1]])
    return [[int(chunks[i]), int(chunks[i+1])] for i in range(len(chunks)-1)]

def chunk_array_of_indices(indices, chunksize=60000, min_chunksize=512):
    '''
    Alternative chunking. Pass the actual indices or slice that should be chunked.
    
    Useful if accessing only some parts of an array by chunks, but slower than chunk_indices()
    
    chunked_indices = chunk_array_slice(indices)
    
    Max Melin - Nov 2023
    '''
    assert min_chunksize < chunksize, "Minimum chunksize must be less than desired chunksize"
    chunks = []
    for c in range(len(indices) // chunksize):
        inds2get = indices[c*chunksize : (c+1)*chunksize]
        if len(inds2get):
            chunks.append(inds2get.tolist())
    chunks.append(indices[(c+1)*chunksize:]) #tack on the last chunk
                              
    if len(chunks[-1]) < min_chunksize: #if last chunk is too small, append it to the previous chunk
       chunks[-2].extend(chunks[-1]) 
       chunks = chunks[:-1]
    return chunks


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



from scipy.interpolate import interp2d
from scipy.signal import ellip, filtfilt,butter

def bandpass_filter(X, sampling_rate, lowpass = 300, highpass = 12000, order=3, method = 'butter'):
    # see also bandpass_filter_gpu in .raw 
    if method == 'ellip':
        b, a = ellip(order, 0.1, 40, 
                     np.array([lowpass,highpass])/(sampling_rate/2.),
                     btype='bandpass')
    elif method == 'butter':
        b, a = butter(order, 
                     np.array([lowpass,highpass])/(sampling_rate/2.),
                     btype='bandpass')
    return filtfilt(b, a, X,axis = 0)

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
        print(f'Discarding {np.sum(np.isnan(input_array))} NaN''s from input array')  #TODO: print out the specific variable name
        return input_array[~np.isnan(input_array)]
    else:
        return input_array


import h5py
def save_dict_to_h5(filename,dictionary,compression = 'lzf', compression_size_threshold = 1000):
    '''
    Save a dictionary as a compressed hdf5 dataset.
    filename: path to the file (IMPORTANT: this WILL overwrite without checks.)
    dictionary: the dictionary to save

    If the size of the data are larger than compression_size_threshold it will save with compression.
    default compression is lzf (fast but little compression).

    '''
    def _save_dataset(f,key,val,compression = compression, compression_size_threshold = compression_size_threshold):
        # compress if big enough.
        if sys.getsizeof(val)>compression_size_threshold:
                compression = compression
        else:
            compression = None
        f.create_dataset(str(key),data = val,compression=compression)
    
    with h5py.File(filename,'w') as f:
        for k in tqdm(dictionary.keys()):
            if not type(dictionary[k]) in [dict]:
                _save_dataset(f,k,dictionary[k])
            else:
                for o in dictionary[k].keys():
                    _save_dataset(f,k+'/'+str(o),dictionary[k][o])

def load_dict_from_h5(filename):
    ''' Loads a dictionary from hdf5'''
    #TODO: read also attributes.
    data = {}
    with h5py.File(filename,'r') as f:
        for k in f.keys():        
            no = k
            if no.isdigit():
                no = int(k)
            if hasattr(f[k],'dims'):
                data[no] = f[k][()]
            else:
                data[no] = dict()
                for o in f[k].keys(): # is group
                    ko = o
                    if o.isdigit():
                        ko = int(o)
                    data[no][ko] = f[k][o][()]
    return data

def _parinit():
    import os
    os.environ['MKL_NUM_THREADS'] = "1"
    os.environ['OMP_NUM_THREADS'] = "1"

def runpar(f,X,nprocesses = None, silent = True,desc = None,**kwargs):
    ''' 
    res = runpar(function,          # function to execute
                 data,              # data to be passed to the function
                 nprocesses = None, # defaults to the number of cores on the machine
                 **kwargs)          # additional arguments passed to the function (dictionary)

    Joao Couto - wfield, 2020

    '''
    if nprocesses is None:
        nprocesses = cpu_count()
    with Pool(initializer = _parinit, processes=nprocesses) as pool:
        if silent:
            res = imap(partial(f,**kwargs),X)
        else:
            res = []
            for r in tqdm(pool.imap_ordered(partial(f,**kwargs),X),
                          total = len(X),desc = desc):
                res.append(r)
    return res


def shifts_from_adc_channel_groups(adc_channel_groups):
    
    shifts = []
    nsamples = len(adc_channel_groups) 
    for i,group in enumerate(adc_channel_groups):
        shifts.append(np.ones(len(group))*i/nsamples)
    channel_ids,tshifts = np.hstack(adc_channel_groups),np.hstack(shifts)
    isort = np.argsort(channel_ids)
    return channel_ids[isort],tshifts[isort]

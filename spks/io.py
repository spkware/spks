from .utils import *
from natsort import natsorted
import numpy as np 
import pandas as pd
import os
import tqdm as tqdm

def is_phy_curated(*sortfolders):
    """takes arbitrary number of spike sorting result folders (need to unpack list when calling) and will return a list
    of wether those results have been curated in phy or not. Will also accept a single string and return a boolean."""
    #TODO: verify this works when curation has been done
    is_curated = []
    for folder in sortfolders:
        ksdata = pd.read_csv(Path(folder) / 'cluster_KSLabel.tsv', sep='\t',header=0)
        phydata = pd.read_csv(Path(folder) / 'cluster_group.tsv', sep='\t', header=0)
        is_curated.append(not ksdata.equals(phydata))
    if len(is_curated) == 1:
        is_curated = is_curated[0]
    return is_curated

def list_spikeglx_binary_paths(subject_dir, return_dates=False):
    """return a list of spikeglx files present for each probe"""
    #TODO: check for missing probe data
    #TODO: add is_sorted flag to only grab sorted sessions
    bin_paths = list(Path(subject_dir).expanduser().glob('**/*.ap.bin'))
    probe_names = natsorted(set([path.name for path in bin_paths])) #assumes the last folder in the path defines the probe name

    all_probe_dirs = []
    for probe_name in probe_names:
        probe_dirs = [str(folder) for folder in bin_paths if folder.name == probe_name]
        probe_dirs = natsorted(probe_dirs, key=str)
        all_probe_dirs.append(probe_dirs)
    return all_probe_dirs

def list_kilosort_result_paths(subject_dir, return_dates=False):
    """return a list of spike-sorting paths for each probe"""
    #TODO: add has_phy flag to only grab curated sessions
    #TODO: check for missing probe data
    ks_dirs = Path(subject_dir).expanduser().glob('**/spike_times.npy')
    ks_dirs = [folder.parent for folder in ks_dirs]
    #if has_phy:
    #    ks_dirs = [folder for folder in ks_dirs if (folder/'.phy').is_dir()]
    probe_names = natsorted(set([path.name for path in ks_dirs])) #assumes the last folder in the path defines the probe name
    all_probe_dirs = []
    for probe_name in probe_names:
        probe_dirs = [str(folder) for folder in ks_dirs if folder.name == probe_name]
        probe_dirs = natsorted(probe_dirs, key=str)
        all_probe_dirs.append(probe_dirs)
    if not return_dates:
        return all_probe_dirs
    else:
        from dateparser.search import search_dates
        dates = [search_dates(str(folder), languages=['en']) for folder in all_probe_dirs[0]] #FIXME: currently returns none

        return all_probe_dirs, dates

def map_binary(fname,nchannels,dtype=np.int16,
               offset = 0,
               mode = 'r',nsamples = None,transpose = False):
    ''' 
    dat = map_binary(fname,nchannels,dtype=np.int16,mode = 'r',nsamples = None)
    
Memory maps a binary file to numpy array.
    Inputs: 
        fname           : path to the file
        nchannels       : number of channels
        dtype (int16)   : datatype
        mode ('r')      : mode to open file ('w' - overwrites/creates; 'a' - allows overwriting samples)
        nsamples (None) : number of samples (if None - gets nsamples from the filesize, nchannels and dtype)
    Outputs:
        data            : numpy.memmap object (nchannels x nsamples array)
See also: map_spikeglx, numpy.memmap

    Usage:
Plot a chunk of data:
    dat = map_binary(filename, nchannels = 385)
    chunk = dat[:-150,3000:6000]
    
    import pylab as plt
    offset = 40
    fig = plt.figure(figsize=(10,13)); fig.add_axes([0,0,1,1])
    plt.plot(chunk.T - np.nanmedian(chunk,axis = 1) + offset * np.arange(chunk.shape[0]), lw = 0.5 ,color = 'k');
    plt.axis('tight');plt.axis('off');

    '''
    dt = np.dtype(dtype)
    if not os.path.exists(fname):
        if not mode == 'w':
            raise(ValueError('File '+ fname +' does not exist?'))
        else:
            print('Does not exist, will create [{0}].'.format(fname))
            if not os.path.isdir(os.path.dirname(fname)):
                os.makedirs(os.path.dirname(fname))
    if nsamples is None:
        if not os.path.exists(fname):
            raise(ValueError('Need nsamples to create new file.'))
        # Get the number of samples from the file size
        nsamples = os.path.getsize(fname)/(nchannels*dt.itemsize)
    ret = np.memmap(fname,
                    mode=mode,
                    dtype=dt,
                    shape = (int(nsamples),int(nchannels)))
    if transpose:
        ret = ret.transpose([1,0])
    return ret

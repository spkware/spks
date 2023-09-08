from .utils import *

def extract_waveform_set(spike_times, data, chmap=None,scratch_directory=None, 
                         max_n_spikes=100,npre=30, npost=30, **extract_waveforms_kwargs):
    """
    
    Take all_listed_timestamps which is a list of the timestamps for each cluster. 
    run extract_waveforms on them (or a subset) and return the mean
    
    scratch_directory: scratch directory if using mmap_output
    Parameters
    ----------
    spike_times : ndarray
        list of timestamps (in samples) of each cluster
    data : Numpy "array-like" (in practice, this is usually np.memmap or spks.raw.RawRecording due to the size of the array)
        absolute path to the binary file 
    scratch_directory : string or Path
        Temporary folder for saving the memory-mapped waveforms. This should be the fastest drive availible on the computer.
    chmap : _type_, optional
        order of channels to read from file, by default None
    chunksize : int, optional
        chunk_size for parallel processing, by default 100
    npre : int, optional
        number of samples before a spike to grab, by default 30
    npost : int, optional
        number of samples after a spike to grab, by default 30
    
    Returns
    -------
    list of waveforms per cluster or TemporaryArrayOnDisk
        An extension of np.memmap that will automatically delete the binary file when the variable goes out of scope or is deleted.
        Size: (n_timestamps, npre+npost, nchannels) and can be indexed like a numpy array.
        Deletion will not work upon a forced exit.

    NOTES: 

    1) If data are in a fast drive (NVME), it will take around 30s to extract 1000 waveforms from a 2h recording (15min in a slow drive).
    2) This function will not consider truncated waveforms that may occur at the edges of the recording (beginning or end).


    Max Melin -spks, 2023
    """

    #print(f'Extracting mean waveforms with up to {max_n_spikes} spikes per cluster.')
    from .max_waveforms import extract_memmapped_waveforms
    all_waves = []
    for s in tqdm(spike_times):
        times_to_extract = np.sort(np.random.choice(s[(s>npre) & (s<(data.shape[0]-npost))].astype(int), size=min(s.size,max_n_spikes), replace=False))
        waves = extract_memmapped_waveforms(data = data, timestamps = times_to_extract, scratch_directory = scratch_directory, 
                                            chmap = chmap, npre=npre, npost=npost, silent=True, **dict(extract_waveforms_kwargs))
        all_waves.append(waves)
    return all_waves


def waveforms_position(waveforms,channel_positions):
    ''' 
    waveforms [ncluster x nsamples x nchannels]
    '''
    peak_to_peak = (waveforms.max(axis = 1) - waveforms.min(axis = 1))
    # the amplitude of each waveform is the max of the peak difference for all channels
    amplitude = np.abs(peak_to_peak).max(axis=1)
    # compute the center of mass (X,Y) of the templates
    centerofmass = [peak_to_peak*pos for pos in channel_positions.T]
    centerofmass = np.vstack([np.sum(t,axis =1 )/np.sum(peak_to_peak,axis = 1) 
                                        for t in centerofmass]).T
    peak_channels = np.argmax(np.abs(peak_to_peak),axis = 1)
    return centerofmass,peak_channels

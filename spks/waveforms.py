from .utils import *

########################################################
##########WAVEFORM METRICS AND ANALYSIS#################
########################################################

def waveforms_position(waveforms,channel_positions):
    ''' 
    Calculates the position of a unit in a set of channels using the center of mass.
    TODO: Add support for other ways if calculating the position... 
    Or maybe wait until there is a case where this doesn't work?

    centerofmass,peak_channels = waveforms_position(waveforms,channel_positions)

    Inputs
    ------
    waveforms : array [ncluster x nsamples x nchannels]
        average waveform for a cluster 
    channel_positions : array [nchannels x 2]
        x and y coordinates of each channel
    
    Returns
    -------
    centerofmass: array [nchannels x 2]
        center of mass of the waveforms 
    peak_channels array [nchannels x 1]
        peak channel of the waveform (the argmax of the absolute amplitude)

    Joao Couto - spks 2023
    '''
    peak_to_peak = (waveforms.max(axis = 1) - waveforms.min(axis = 1))
    # the amplitude of each waveform is the max of the peak difference for all channels
    amplitude = np.max(peak_to_peak,axis=1) 
    # compute the center of mass (X,Y) of the waveforms
    centerofmass = [peak_to_peak*pos for pos in channel_positions.T]
    centerofmass = np.vstack([np.sum(t,axis =1 )/np.sum(peak_to_peak,axis = 1) 
                                        for t in centerofmass]).T
    # the peak channel is the index of the channel that has the largest deflection
    peak_channels = np.argmax(np.abs(waveforms).max(axis = 1), axis = 1)

    return centerofmass, peak_channels

def compute_waveform_metrics(waveform,npre,srate,upsampling_factor = 100):
    '''
    Computes the spike waveform metrics for a single waveform

    wavemetrics = waveforms_position(waveform,npre,srate,upsample_factor = 100)

    Inputs
    ------
    waveforms : array [nsamples]
        average waveform for a single channel (usually the channel with the biggest waveform) 
    npre : int
        number of samples taken before the spike
    srate: float
        sampling rate used to sample the data
    upsampling_factor: int (default 100)
        factor to 'upsample' the waveform to avoid sampling artifacts in the quantification

    Returns
    -------
    wavemetrics: dict 
        trough_time: (ms) time to the trough in relation to the spike timestamp
        trough_amplitude: trough amplitude
        fw3m: full-witdh at 2/3 maximum 
        trough_gradient: gradient at 0.07 from the trough  
        peak_gradient: gradient at 0.5ms from the trough
        peak_time: time to the peak in relation to the spike timestamp
        peak_amplitude: amplitude of the peak
        spike_duration: time between the trough and the peak
        polarity:  -1 for negative waveforms, 1 for positive

    Joao Couto - spks 2023

    Usage example
    -------------
    # get the principal waveforms from Clusters
    principal_waveforms = np.stack([w[:,p] for w,p in zip(clusters.cluster_waveforms_mean,clusters.cluster_channel)])

    srate = 30000
    npre = 30
    # compute the waveform of each cluster
    clumetrics = []
    for w in tqdm(principal_waveforms):
        clumetrics.append( compute_waveform_metrics(w,npre=30,srate=srate) )
    clumetrics = pd.DataFrame(clumetrics)

    iclu = 4
    # plot a waveform with the trough and peak locations
    plt.figure()
    t = np.arange(-npre,len(waveform)-npre)/srate
    plt.plot(t*1000,principal_waveforms[iclu].T,'k');

    plt.plot(clumetrics.trough_time.iloc[iclu],clumetrics.trough_amplitude.iloc[iclu],'bo')
    plt.plot(clumetrics.peak_time.iloc[iclu],clumetrics.peak_amplitude.iloc[iclu],'bo')


    '''
    t = np.arange(-npre,len(waveform)-npre)/srate

    from scipy.interpolate import interp1d
    # interpolate to a higher temporal resolution because spikes are fast, so we don't get digitized metrics..
    spline_interp = interp1d(t,waveform, kind='quadratic') 
    tt = np.linspace(t[0],t[-1],int((t[-1]-t[0])*srate*upsampling_factor))
    wv = spline_interp(tt)
    tt *= 1000 # in ms
    # to estimate the trough we check the absolute maxima of the waveform (0.1 ms) around zero
    # this will work also if the waveform is positive (but not well tested)
    # it estimates the trough time, amplitude and gradient
    # the gradient is 
    trough_idx = np.argmax(np.abs(wv[(tt>=-0.25) & (tt<=0.25)])) + np.where((tt>=-0.25))[0][0]
    trough_amp = wv[trough_idx]
    trough_time = tt[trough_idx]
    
    # full-width at 2/3 maximum is the width of the spike at 2/3 of the maxima (lets consider the maxima the trought)
    # lets again work in abs to try to make this more general.
    fw3m_idx1 = np.where(np.abs(wv[(tt<=trough_time)])<=np.abs(2*trough_amp/3))[0] # where it intersects before zero
    fw3m_idx2 = np.where(np.abs(wv[(tt>=trough_time)])<=np.abs(2*trough_amp/3))[0] # where it intersects after zero
    if not len(fw3m_idx1) or not len(fw3m_idx2):
        fw3m = np.nan  # this happens for artifacts..
        print('[waveform metrics] - Waveform has no full-width at 2/3 maximum')
    else:
        fw3m_idx1 = fw3m_idx1[-1]  # we only need the last index
        fw3m_idx2 = fw3m_idx2[0] + np.where(tt>=trough_time)[0][0]+1 # we only need the first index plus the offset
        fw3m =  tt[fw3m_idx2] - tt[fw3m_idx1]

    # trought gradient 
    # the trought gradient is computed 0.07ms from the trought (as per I-Chun et al. 2020)
    idx = np.where(tt>=(trough_time+0.07))[0][0]
    # the gradient is in uV/ms (if the waveform is in uV)
    trough_gradient = (wv[idx+1] - wv[idx])/(tt[idx+1]-tt[idx])

    # peak gradient 
    # the peak gradient is computed 0.5ms from the trought (as per I-Chun et al. 2020)
    idx = np.where(tt>=(trough_time+0.5))[0][0]
    # the gradient is in uV/ms (if the waveform is in uV)
    peak_gradient = (wv[idx+1] - wv[idx])/(tt[idx+1]-tt[idx])

    # the peak is at abs max after the trought and after crossing zero
    # we compute the peak to be able to compute the amplitude and the spike duration
    idx = (np.where(wv[trough_idx:]>=0)[0][0])+trough_idx
    peak_idx = np.argmax(wv[idx:])+idx
    peak_time = tt[peak_idx]
    peak_amp = wv[peak_idx]
    # spike duration is the time between trough and the peak
    spike_duration = peak_time - trough_time
    if spike_duration == 0: # then it is an artifact
        spike_duration = np.nan
    wavemetrics = dict(trough_time = trough_time,
                    trough_amplitude  = trough_amp,
                    fw3m = fw3m,
                    trough_gradient = trough_gradient,
                    peak_gradient = peak_gradient,
                    peak_time = peak_time,
                    peak_amplitude  = peak_amp,
                    spike_duration = spike_duration,
                    polarity = int(-1 if trough_amp<0 else 1))  #sign of the through is the polarity
    return wavemetrics


def estimate_active_channels(cluster_waveforms_mean,madthresh = 12):
    '''
    TODO
    '''
    nclusters,nsamples,nchannels = cluster_waveforms_mean.shape

    peak_amp = [mwave[nsamples//2-15:nsamples//2+15,:].max(axis=0) - mwave[nsamples//2-15:nsamples//2+15,:].min(axis=0)
            for mwave in cluster_waveforms_mean]
    # get the threshold from the median_abs_deviation, this is quite low and we are comparing with the peak amplitude so use a high madthresh
    mad_thresh = mad(cluster_waveforms_mean[:,:nsamples//2-15,:].reshape(-1,nchannels)) 

    # "active" channels
    activeidx = []
    for p in peak_amp:
            activeidx.append(np.where(p>mad_thresh*madthresh)[0])

    nactive_channels = np.array([len(a) for a in activeidx])
    return activeidx


########################################################
################EXTRACT WAVEFORMS#######################
########################################################


def __work_extract_waveforms(data, waveforms, timestamps, time_indices, chmap, chunk_inds, flush_memory):
    """Extracts waveforms from binary file and writes them to the global variable waveforms."""
    
    spike_times = timestamps.flatten()[chunk_inds]

    holder = np.empty((len(chunk_inds),len(time_indices),len(chmap)))
    for i,s in enumerate(spike_times):
        holder[i,:,:] = np.take(data[time_indices+s,:].astype(np.int16),chmap,axis=1)
    waveforms[chunk_inds,:,:] = holder

    if flush_memory:
        waveforms.flush() #runs MUCH faster with no flush if sufficient memory, but no flush is much slower if memory is exceeded, which it usually is
    return

def extract_memmapped_waveforms(data, scratch_directory, timestamps, mmap_output=False, flush_memory=True, silent=False, chunksize=10, npre=30, npost=30, chmap=None):
    """Takes an array of timestamps and extracts the waveforms on all channels. Waveforms are memory mapped to a binary
    file to overcome memory limits.

    Parameters
    ----------
    data : Numpy "array-like" (in practice, this is usually np.memmap or spks.raw.RawRecording due to the size of the array)
        absolute path to the binary file 
    scratch_directory : string or Path
        Temporary folder for saving the memory-mapped waveforms. This should be the fastest drive availible on the computer.
    timestamps : ndarray
        the timestamps (in samples) of each spike to be extracted
    chunksize : int, optional
        chunk_size for parallel processing, by default 10
    npre : int, optional
        number of samples before a spike to grab, by default 30
    npost : int, optional
        number of samples after a spike to grab, by default 30
    chmap : _type_, optional
        _description_, by default None

    Returns
    -------
    TemporaryArrayOnDisk
        An extension of np.memmap that will automatically delete the binary file when the variable goes out of scope or is deleted.
        Size: (n_timestamps, npre+npost, nchannels) and can be indexed like a numpy array.
        Deletion will not work upon a forced exit.

    Max Melin - 2023
    """

    if chmap is None:
        chmap = np.arange(data.shape[1])
    
    nchannels = len(chmap)

    n_chunks = timestamps.size // chunksize + 1
    chunks = np.arange(n_chunks)
    time_indices = np.arange(-npre,npost,dtype=np.int16)

    chunk_inds = []
    for c in chunks:
        inds2get = np.arange(c*chunksize, (c+1)*chunksize)
        inds2get = inds2get[inds2get < timestamps.size] #truncate last chunk
        if len(inds2get):
            chunk_inds.append(inds2get)

    mmap_shape = (len(timestamps),npre+npost,nchannels)
    if mmap_output:
        assert not scratch_directory is None, "[extract_memmapped_waveforms] - must specify scratch directory if mmap_output is True."
        tfile = TemporaryArrayOnDisk(scratch_directory,
                                     mode='w+',
                                     dtype=np.int16, 
                                     order='C',
                                     shape = mmap_shape)
        tfile.flush()
        if not silent:
            print(f'\nWaveforms mapped to {tfile.filename}')
    else:
        tfile = np.zeros(shape=mmap_shape, dtype=np.int16)
        flush_memory = False

    mpfunc = partial(__work_extract_waveforms, data, tfile, timestamps, time_indices, chmap, flush_memory=flush_memory)

    with ThreadPool() as pool:
            if not silent:
                print(f'Extracting waveforms with chunk-size {chunksize}')
                for _ in tqdm(pool.imap_unordered(mpfunc, chunk_inds), desc = 'Extracting waveforms', total=len(chunk_inds)):
                    pass
            else:
                for _ in pool.imap_unordered(mpfunc, chunk_inds): # no waitbar
                    pass
                
    return tfile

def extract_waveform_set(spike_times, data, chmap=None,scratch_directory=None, 
                         max_n_spikes=1000,npre=30, npost=30, **extract_waveforms_kwargs):
    """
    
    Take a list of spike_times which is a list of the timestamps for each cluster. 
    run extract_memmapped_waveforms on and return as a list of arrays or memmapped arrays.
    
    scratch_directory: scratch directory if using mmap_output
    Parameters
    ----------
    spike_times : list
        list of timestamps (in samples) of each cluster
    data : Numpy "array-like" (in practice, this is usually np.memmap or spks.raw.RawRecording due to the size of the array)
        absolute path to the binary file 
    scratch_directory : string or Path
        Temporary folder for saving the memory-mapped waveforms. This should be the fastest drive availible on the computer.
    chmap : _type_, optional
        order of channels to read from file, by default None
    max_n_spikes : int, optional
        the number of randomly selected waveforms to grab for each set of spikes in spike_times, by default 1000
    chunksize : int, optional
        chunk_size for parallel processing, by default 100
    npre : int, optional
        number of samples before a spike to grab, by default 30
    npost : int, optional
        number of samples after a spike to grab, by default 30
    
    Returns
    -------
    list of waveforms per cluster. The items in this list are ndarray or TemporaryArrayOnDisk.
        An extension of np.memmap that will automatically delete the binary file when the variable goes out of scope or is deleted.
        Size: (n_timestamps, npre+npost, nchannels) and can be indexed like a numpy array.
        Deletion will not work upon a forced exit.

    NOTES: 

    1) If data are in a fast drive (NVME), it will take around 30s to extract 1000 waveforms from a 2h recording (15min in a slow drive).
    2) This function will not consider truncated waveforms that may occur at the edges of the recording (beginning or end).


    Max Melin - spks, 2023
    """

    #print(f'Extracting mean waveforms with up to {max_n_spikes} spikes per cluster.')
    all_waves = []
    for s in tqdm(spike_times,desc = 'Extracting waveforms'):
        times_to_extract = np.sort(np.random.choice(s[(s>npre) & (s<(data.shape[0]-npost))].astype(int), size=min(s.size,max_n_spikes), replace=False))
        waves = extract_memmapped_waveforms(data = data, timestamps = times_to_extract, scratch_directory = scratch_directory, 
                                            chmap = chmap, npre=npre, npost=npost, silent=True, **dict(extract_waveforms_kwargs))
        all_waves.append(waves)
    return all_waves



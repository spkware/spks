from .utils import *

########################################################
##########WAVEFORM METRICS AND ANALYSIS#################
########################################################

def channel_neighbours(channel_positions, max_gap = 1):
    '''
    Which channels of a probe are next to each other, for any layout of the channels.

    Two channels are neighbours when they share an edge of the Delaunay
    triangulation of the channel positions (their natural neighbours), so the
    neighbours follow the layout even when the spacing is irregular.  Edges over
    3 times the median distance between nearest channels are left out, so
    channels on different shanks or across a gap in the layout are not
    neighbours.  With max_gap, channels with up to max_gap channels between them
    are also neighbours.

    neighbours = channel_neighbours(channel_positions, max_gap = 1)

    Inputs
    ------
    channel_positions : array [nchannels x 2]
        x and y coordinates of each channel (um)
    max_gap : int
        number of channels allowed between two neighbours (default 1)

    Return
    -------
    neighbours : bool array [nchannels x nchannels]
        True for neighbours (and on the diagonal)
    '''
    from scipy.spatial import Delaunay
    from scipy.spatial.distance import cdist
    positions = np.asarray(channel_positions, dtype = float)
    nchannels = len(positions)
    distance = cdist(positions, positions)
    edges = np.zeros((nchannels, nchannels), dtype = bool)
    if nchannels > 1:
        try:
            simplices = Delaunay(positions).simplices
            for a, b in ((0, 1), (0, 2), (1, 2)):
                edges[simplices[:, a], simplices[:, b]] = True
        except Exception:
            # all channels on one line (or fewer than 3): link them in order along the line
            centered = positions - positions.mean(axis = 0)
            direction = np.linalg.svd(centered, full_matrices = False)[2][0]
            order = np.argsort(centered @ direction)
            edges[order[:-1], order[1:]] = True
        edges |= edges.T
        nearest = np.where(distance > 0, distance, np.inf).min(axis = 1)
        nearest = nearest[np.isfinite(nearest)]
        if len(nearest):
            edges &= distance <= 3*np.median(nearest)
        edges |= distance == 0   # channels at the same position
    neighbours = edges | np.eye(nchannels, dtype = bool)
    steps = edges.astype(np.float32)
    for _ in range(int(max_gap)):  # one more channel between neighbours
        neighbours = neighbours | ((neighbours.astype(np.float32) @ steps) > 0)
    return neighbours

def waveforms_footprint(waveforms, channel_positions, threshold = 0.2, max_gap = 1):
    '''
    The channels each waveform spans, its position and its extent on the probe.

    The active electrodes of a waveform are the channels with a peak-to-peak of
    at least threshold (20%) of the peak-to-peak on the peak channel, reached
    from the peak channel through neighbouring channels that also pass, allowing
    max_gap (1) channels that do not pass in between (see channel_neighbours).
    The threshold is relative, so it does not depend on the amplitude or the
    noise, and a waveform that is about as large on every channel (e.g.
    electrical noise) spans every channel of the shank.  Channels that pass but
    are far from the peak (e.g. a broken channel) are not active.

    center_of_mass, peak_channels, active_electrodes, span_um = waveforms_footprint(waveforms, channel_positions)

    Inputs
    ------
    waveforms : array [nclusters x nsamples x nchannels]
        average or median waveform of each cluster
    channel_positions : array [nchannels x 2]
        x and y coordinates of each channel (um)
    threshold : float
        fraction of the peak-to-peak of the peak channel (default 0.2)
    max_gap : int
        number of channels below threshold allowed in between (default 1)

    Return
    -------
    center_of_mass : array [nclusters x 2]
        position of each waveform: the center of the active electrodes,
        weighted by the peak-to-peak (NaN for empty waveforms)
    peak_channels : array [nclusters]
        channel with the largest peak-to-peak
    active_electrodes : list of arrays
        channel indices of the active electrodes (empty for empty waveforms)
    span_um : array [nclusters]
        largest distance between two active electrodes (um); 0 for one channel

    Joao Couto - spks 2023
    '''
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import breadth_first_order
    waveforms = np.asarray(waveforms)
    positions = np.asarray(channel_positions, dtype = float)
    neighbours = channel_neighbours(positions, max_gap = max_gap)
    peak_to_peak = np.nan_to_num(waveforms.max(axis = 1) - waveforms.min(axis = 1), nan = 0.)
    center_of_mass, peak_channels, active_electrodes, span_um = [], [], [], []
    for ptp in peak_to_peak:
        peak = int(np.argmax(ptp)) # the peak channel is the index of the channel that has the largest deflection
        peak_channels.append(peak)
        if not ptp[peak] > 0:  # an empty waveform
            active_electrodes.append(np.zeros(0, dtype = int))
            center_of_mass.append([np.nan]*2)
            span_um.append(np.nan)
            continue
        passing = np.where(ptp >= threshold*ptp[peak])[0]
        reached = breadth_first_order(csr_matrix(neighbours[np.ix_(passing, passing)]),
                                      int(np.searchsorted(passing, peak)),
                                      directed = False, return_predecessors = False)
        active = np.sort(passing[reached])
        active_electrodes.append(active)
        weights = ptp[active]
        center_of_mass.append(np.sum(weights[:, None]*positions[active], axis = 0)/np.sum(weights))
        extent = positions[active]
        span_um.append(float(np.max(np.linalg.norm(extent[:, None] - extent[None], axis = -1))))
    return np.array(center_of_mass), np.array(peak_channels), active_electrodes, np.array(span_um)

def waveforms_position(waveforms, channel_positions, threshold = 0.2, max_gap = 1):
    '''
    Position, peak channel and active electrodes of each waveform.

    The center of mass of the active electrodes; see waveforms_footprint, which
    also gives the extent (span_um) of each waveform.

    center_of_mass, peak_channels, active_electrodes = waveforms_position(waveforms, channel_positions)

    Inputs
    ------
    waveforms : array [nclusters x nsamples x nchannels]
        average or median waveform of each cluster
    channel_positions : array [nchannels x 2]
        x and y coordinates of each channel (um)
    threshold : float
        fraction of the peak-to-peak of the peak channel (default 0.2)
    max_gap : int
        number of channels below threshold allowed in between (default 1)

    Return
    -------
    center_of_mass : array [nclusters x 2]
    peak_channels : array [nclusters]
    active_electrodes : list of arrays

    Joao Couto - spks 2023
    '''
    center_of_mass, peak_channels, active_electrodes, _ = waveforms_footprint(
        waveforms, channel_positions, threshold = threshold, max_gap = max_gap)
    return center_of_mass, peak_channels, active_electrodes

def compute_waveform_metrics(waveform,npre,srate,upsampling_factor = 100, trough_window = 0.5):
    '''
    Computes the spike waveform metrics for a single waveform

    wavemetrics = compute_waveform_metrics(waveform,npre,srate,upsample_factor = 100)

    Parameters
    ------------

    waveforms : array [nsamples]
        average waveform for a single channel (usually the channel with the biggest waveform) 
    npre : int
        number of samples taken before the spike
    srate: float
        sampling rate used to sample the data
    upsampling_factor: int (default 100)
        factor to 'upsample' the waveform to avoid sampling artifacts in the quantification
    trough_window: float (default 0.5)
        the trough is the largest absolute deflection within this many ms of the spike
        timestamp; sorters do not always align the spike time to the trough

    Return
    ------------

    wavemetrics: dict
        trough_time: (ms) time to the trough in relation to the spike timestamp
        trough_amplitude: trough amplitude (the largest deflection, positive for positive waveforms)
        fw3m: full-witdh at 2/3 maximum
        trough_gradient: gradient at 0.07 ms from the trough, NaN past the end of the waveform
        peak_gradient: gradient at 0.5 ms from the trough, NaN past the end of the waveform
        peak_time: time to the peak in relation to the spike timestamp
        peak_amplitude: amplitude of the peak, the extremum of opposite sign after the trough
        spike_duration: time between the trough and the peak
        peak_to_peak: difference between the maximum and the minimum of the waveform
        polarity:  -1 for negative waveforms, 1 for positive

    Joao Couto - spks 2023

    Example
    ------------

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
    # the trough is the absolute maxima of the waveform within trough_window (ms) around zero
    # its sign is the polarity, so this works also if the waveform is positive
    window = np.where((tt>=-trough_window) & (tt<=trough_window))[0]
    trough_idx = window[np.argmax(np.abs(wv[window]))]
    trough_amp = wv[trough_idx]
    trough_time = tt[trough_idx]
    polarity = int(-1 if trough_amp<0 else 1)  # sign of the trough is the polarity

    def gradient_at(time):
        # gradient in uV/ms (if the waveform is in uV), NaN past the end of the waveform
        idx = np.where(tt>=time)[0]
        if len(idx) < 2:
            return np.nan
        idx = idx[0]
        return (wv[idx+1] - wv[idx])/(tt[idx+1]-tt[idx])

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
    trough_gradient = gradient_at(trough_time+0.07)

    # peak gradient
    # the peak gradient is computed 0.5ms from the trought (as per I-Chun et al. 2020)
    peak_gradient = gradient_at(trough_time+0.5)

    # the peak is the extremum of opposite sign after the trough, once the waveform crosses zero
    # (the maximum for negative waveforms, the minimum for positive ones)
    # we compute the peak to be able to compute the amplitude and the spike duration
    flipped = -polarity*wv  # the peak is a maximum of the flipped waveform
    idx = np.where(flipped[trough_idx:]>=0)[0]
    if len(idx):
        idx = idx[0]+trough_idx
        peak_idx = np.argmax(flipped[idx:])+idx
        peak_time = tt[peak_idx]
        peak_amp = wv[peak_idx]
        # spike duration is the time between trough and the peak
        spike_duration = peak_time - trough_time
        if spike_duration == 0: # then it is an artifact
            spike_duration = np.nan
    else:
        # there is no peak
        peak_time = np.nan
        peak_amp = np.nan
        spike_duration = np.nan
    wavemetrics = dict(trough_time = trough_time,
                    trough_amplitude  = trough_amp,
                    fw3m = fw3m,
                    trough_gradient = trough_gradient,
                    peak_gradient = peak_gradient,
                    peak_time = peak_time,
                    peak_amplitude  = peak_amp,
                    spike_duration = spike_duration,
                    peak_to_peak = wv.max() - wv.min(),
                    polarity = polarity)
    return wavemetrics


def estimate_active_channels(cluster_waveforms_mean, channel_positions, threshold = 0.2, max_gap = 1):
    '''
    The active electrodes of each waveform; see waveforms_footprint.

    activeidx = estimate_active_channels(cluster_waveforms_mean, channel_positions)
    '''
    return waveforms_footprint(cluster_waveforms_mean, channel_positions,
                               threshold = threshold, max_gap = max_gap)[2]


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
                         max_n_spikes=1000,npre=45, npost=45, **extract_waveforms_kwargs):
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
        valid_spikes = s[(s>npre) & (s<(data.shape[0]-npost))].astype(int) #exclude waveforms truncated by the end or beginning of recording
        times_to_extract = np.sort(np.random.choice(valid_spikes, size=min(valid_spikes.size,max_n_spikes), replace=False))
        waves = extract_memmapped_waveforms(data = data, timestamps = times_to_extract, scratch_directory = scratch_directory, 
                                            chmap = chmap, npre=npre, npost=npost, silent=True, **dict(extract_waveforms_kwargs))
        all_waves.append(waves)
    return all_waves



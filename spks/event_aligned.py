from .utils import *
from collections.abc import Iterable

def get_triggered_unit_spikes(ts,events,tpre = 1,tpost = 1):
    trig = []
    for e in events:
        trig.append(ts[(ts> (e-tpre)) &(ts< (e+tpost))]-e)
    return trig

def get_triggered_spikes(units,events,tpre,tpost,ncpus = 14):
    with Pool(ncpus) as pool:
        result = [r for r in tqdm(pool.imap(partial(get_triggered_unit_spikes,
                    events = events,#np.sort(np.hstack([interpfunction(onsets[1]),interpfunction(offsets[1])]))/srate,#,
                    tpre = tpre, tpost = tpost),units),
                desc='Getting triggered spikes', total = len(units))]
    return result

from .viz import plt
def plot_raster(spks, offset=0.0, height=1.0,
                colors='k',
                ax = None,
                mode = 'scatter',
                lw = 0.5,
                markersize = 2, **kwargs):
    ''' Plot a raster from sets of spiketrains.
            - spks: is a list of spiketrains
            - mode: can be scatter (default) or line. Lines creates a line of height defined by 'height'
        Line mode can be used for exporting to pdf and editing more intuitively
            - height
            - "colors" can be an list of colors (per trial - in line mode)
        Joao Couto - January 2016
    '''
    import pylab as plt
    if ax is None:
        ax = plt.gca()
    nspks = len(spks)
    if mode == 'line': # for exporting to pdf and editing
        if type(colors) is str:
            colors = [colors]*nspks
        for i,(sp,cc) in enumerate(zip(spks,colors)):
            ax.vlines(sp,offset+(i*height),
                    offset+((i+1)*height),
                    colors=cc,**kwargs)
    else: # scatter
        a = np.hstack([i*np.ones_like(s) + offset for i,s in enumerate(spks)])
        b = np.hstack(spks)
        if len(colors) == len(spks):
            colors = np.hstack([[colors[i]]*len(s) for i,s in enumerate(spks) if len(s)]) # to color each trial differently
        s = np.ones_like(b)*markersize
        ax.scatter(b,a,s,c = colors, marker = '|', lw = lw, **kwargs)
        ax.autoscale(tight = True)
    
def align_raster_to_event(event_times, spike_times, pre_seconds, post_seconds):
    """create aligned rasters relative to event_times

    Parameters
    ----------
    event_times : list or ndarray
        a list or numpy array of event times to be aligned to
    spike_times : list or ndarray
        a list spike times for one cluster
    pre_seconds : float, list
        grab _ seconds before event_times for alignment, by default 1
        can also be a list of different pre_seconds for each event
    post_seconds : float, list
        grab _ seconds after event_times for alignment, by default 2
        can also be a list of different pre_seconds for each event

    Returns
    -------
    list
        a list of aligned rasters for each event_times
    """    
    event_rasters = []
    pre_iterable = isinstance(pre_seconds, Iterable)
    post_iterable = isinstance(post_seconds, Iterable)
    for i, event_time in enumerate(event_times):
        relative_spiketimes = spike_times - event_time
        pre = pre_seconds[i] if pre_iterable else pre_seconds
        post = post_seconds[i] if post_iterable else post_seconds
        spks = relative_spiketimes[np.logical_and(relative_spiketimes <= post, relative_spiketimes >= -pre)]
        event_rasters.append(np.array(spks))
    return event_rasters

def compute_spike_count(event_times, spike_times, pre_seconds, post_seconds, binwidth_ms=25, kernel=None):
    '''compute the PETH for one neuron'''
    binwidth_s = binwidth_ms/1000
    event_times = discard_nans(event_times) 
    
    rasters = align_raster_to_event(event_times, 
                                spike_times,
                                pre_seconds,
                                post_seconds)
    pre_event_timebins = -np.arange(0, pre_seconds, binwidth_s)[1:][::-1]
    #post_event_timebins = np.arange(0, post_seconds+binwidth_s, binwidth_s)
    post_event_timebins = np.arange(0, post_seconds, binwidth_s)
    timebin_edges = np.append(pre_event_timebins, post_event_timebins)

    event_index = pre_event_timebins.size # index of the alignment event in psth_matrix

    psth_matrix = binary_spikes(rasters, timebin_edges, kernel=kernel) #/ binwidth_s # divide by binwidth to get a rate rather than count
    return psth_matrix, timebin_edges, event_index

def compute_spike_count_truncated(event_times, spike_times, max_pre_seconds, max_post_seconds, pre_seconds, post_seconds, binwidth_ms=25, kernel=None):
    '''similar to compute_spike_count but takes a list of pre and post seconds, as well as a max pre and post time.
    This allows truncation of the psth_matrix, where values that are not within the pre and post time are set to nan'''
    binwidth_s = binwidth_ms/1000
    event_times = discard_nans(event_times)
    rasters = align_raster_to_event(event_times,
                                    spike_times,
                                    pre_seconds,
                                    post_seconds)

    pre_event_timebins = np.arange(-max_pre_seconds, 0, binwidth_s)
    post_event_timebins = np.arange(0, max_post_seconds+binwidth_s, binwidth_s)
    timebin_edges = np.append(pre_event_timebins, post_event_timebins)
    event_index = pre_event_timebins.size # index of the alignment event in psth_matrix
    psth_matrix = np.empty((len(rasters), len(timebin_edges)-1))
    psth_matrix[:] = np.nan
    for i,raster in enumerate(rasters):
        temp = binary_spikes([raster], timebin_edges, kernel=kernel)
        #start_ind = np.where(timebin_edges > -pre_seconds[i])[0][0] - 1
        #stop_ind = np.where(timebin_edges < post_seconds[i])[0][-1]
        aa = timebin_edges > -pre_seconds[i]
        bb = timebin_edges < post_seconds[i]
        if np.any(aa) and np.any(bb):
            start_ind = np.where(aa)[0][0] - 1
            stop_ind = np.where(bb)[0][-1]
            psth_matrix[i, start_ind:stop_ind] = temp[0,start_ind:stop_ind]
        else:
            #print('event outside of time range, skipping trial')
            continue

    return psth_matrix, timebin_edges, event_index
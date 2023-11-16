from .utils import *

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
            colors = np.hstack([[colors[i]]*len(s) for i,s in enumerate(spks)]) # to color each trial differently
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

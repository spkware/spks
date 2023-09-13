import pylab as plt
import numpy as np

colors = ['#000000',
          '#d62728',
          '#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#7f7f7f',
          '#bcbd22',
          '#17becf']

def plot_raster(spks,offset=0.0,height=1.0,colors='black',ax = None,**kwargs):
    ''' Plot a raster from sets of spiketrains.
            - "spks" is a list of spiketrains
            - "colors" can be an array of colors
        Joao Couto - January 2016
    '''
    if ax is None:
        ax = plt.gca()
    nspks = len(spks)
    if type(colors) is str:
        colors = [colors]*nspks
    for i,(sp,cc) in enumerate(zip(spks,colors)):
        ax.vlines(sp,offset+(i*height),
                  offset+((i+1)*height),
                  colors=cc,**kwargs)

def plot_event_aligned_raster(event_times, spike_times, sorting='', pre_seconds=1, post_seconds=2, offset=0, ax=None, color='black', **kwargs):
    """Plot rasters from multiple trials aligned to event_times

    Parameters
    ----------
    event_times : list or ndarray
        a list or numpy array of event times to be aligned to
    spike_times : list or ndarray
        a list of all spike times for one cluster
    sorting : _type_, optional
        _description_, by default None
    pre_seconds : int, optional
        grab _ seconds before event_times for alignment, by default 1
    post_seconds : int, optional
        grab _ seconds after event_times for alignment, by default 2
    ax : _type_, optional
        matplotlib axes object to plot on, by default None and will create an axis
    color : str, optional
        plotting color, by default 'black'
    """    
    #TODO: add subsample option
    event_times = discard_nans(event_times)
    if ax is None:
        ax = plt.gca()
    n_events = len(event_times)

    if sorting == 'time_to_first_spike':
        first_spike_deltas = []
        for event_time in event_times:
            relative_spiketimes = spike_times - event_time
            first_spike_deltas.append(next(time for time in relative_spiketimes if time > 0))
        event_times = event_times[np.argsort(first_spike_deltas)]
        ax.set_ylabel('Sorted by time to first spike')

    rasters = rasterize_to_event(event_times, spike_times, pre_seconds, post_seconds)
    for i, event_raster in enumerate(rasters):
        ax.vlines(event_raster,(offset+i),
                  (offset+i+1), color=color, **kwargs)
    ax.set_xlabel('Time relative to event (s)')
    ax.set_ylim(ax.get_ylim()[::-1]) # flip the y axis

def plot_event_based_raster_fast(event_times, spike_times, pre_seconds=1, post_seconds=2, offset=0, ax=None, color='black', **kwargs):
    """faster version of plot_event_aligned_raster for SpikeViewer"""
    #TODO: add subsample option
    event_times = discard_nans(event_times)
    if ax is None:
        ax = plt.gca()
    n_events = len(event_times)

    rasters = rasterize_to_event(event_times, spike_times, pre_seconds, post_seconds)
    x = []
    ymin =[] 
    ymax = []
    for i, event_raster in enumerate(rasters):
        x.extend(event_raster)
        ymin.extend([offset+i]*len(event_raster))
        ymax.extend([offset+i+1]*len(event_raster))
    x = np.array(x)
    ymin = np.array(ymin)
    ymax = np.array(ymax)
    #ax.vlines(x, ymin, ymax, color=color, **kwargs)
    ax.scatter(x, ymin, s=.2, linewidths=.2, marker='|', color=color, **kwargs) #TODO: adaptive s and linewidth kwargs
    #ax.scatter(x, ymin, s=5, linewidths=5, marker='|', color=color, **kwargs)
    ax.set_xlabel('Time relative to event (s)')
    #ax.set_ylim((np.min(ymin),np.max(ymax)))
    #ax.set_xlim((np.min(x), np.max(x)))
    #ax.set_ylim(ax.get_ylim()[::-1]) # flip the y axis


def plot_multichannel_data(dat, chorder, srate = 30000.,
                           offset = 1000, filterdata = False,
                           removedc = True, colors=None,scalebar=(0.05,100)):
    ''' Plot raw data ordered by channel '''
    dat = np.array(dat[:10000,:],dtype = np.float32)
    time = np.arange(dat.shape[0])/float(srate)
    ax = plt.gca()
    if colors is None:
        colors = [[0,0,0] for i in range(len(chorder))]
    if filterdata:
        # Low pass signal
        y = filter_data(dat[:,chorder.astype(int)],500,0.95*(srate/2),srate)
    else:
        y = dat[:,chorder.astype(int)]
    offsets = np.arange(y.shape[1])*float(offset)
    if removedc:
        offsets -= np.mean(y[0:int(0.1*y.shape[0]),],axis=0)

    y += offsets
    plts = ax.plot(time,y,'k',lw=0.6,clip_on=False)

    for i,p in enumerate(plts):
        p.set_color(colors[i])
    for ii,ch in enumerate(chorder):
        ax.text(time[0],y[0,ii],str(int(ch)),color=[.3,.7,.3],
                fontsize=13,fontweight='bold',va='center',ha='right')
    ax.axis([time[0],time[-1],np.min(y),np.max(y)])
    if not scalebar is None:
        x = time[int(0.05*y.shape[0])]+np.array([0,scalebar[0]])
        y = np.min(y) + np.array([0,0])

        ax.plot(x,y,color='k',clip_on=False)
        ax.text(np.diff(x)/2. + x[0],y[0],'{0} ms'.format(1000*scalebar[0]),
                va = 'top', ha='center')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    return plts,(time,offsets)

def plot_footprints(waves, channel_xy, gain = (3,70),
                                srate = 30000, plotscale=None,color='k',lw=1.):
    '''Plots multichannel waveforms (shape=(nspikes,nsamples,nchannels)) or (shape=(nsamples,nchannels)).
    '''
    import pylab as plt
    wtime = 1000.*np.arange(waves.shape[0])/float(srate)
    p = []
    ax = plt.gca()
    if len(waves.shape) == 2:
        # plotting mean waveforms (these must be average waveforms)
        waves = waves.copy().reshape(1,*waves.shape)
    for wv in waves:
        for i in range(wv.shape[1]):
            # plot each channel
            p.append(ax.plot(wtime*gain[0] + channel_xy[i,0],
                              wv[:,i]*gain[1]+channel_xy[i,1],
                             color=color,
                             lw=lw,clip_on=False))
        miny = np.min(abs(np.diff(channel_xy[:,1])))
        ax.axis([np.min(channel_xy[:,0]),
                 np.max(channel_xy[:,0])+wtime[-1]*gain[0],
                np.min(channel_xy[:,1])-miny,
                np.max(channel_xy[:,1])+miny])
        ax.axis('off')
    if not plotscale is None:
        t = []
        # X scale
        x = plotscale[0] * np.array([0,1])*gain[0] - gain[0]/6.
        y = np.array([0,0])+np.min(channel_xy[i,0]) - gain[1]/2.

        p.append(plt.plot(x,y,'k',clip_on = False,lw = 1))
        t.append(ax.text(np.diff(x)/2. + np.min(x),np.min(y),
                r'{0} ms'.format(plotscale[0]),
                va = 'top',ha='center',fontsize=8))
        # Y scale
        x = np.array([0,0])-gain[0]/6.
        y = np.array([0,plotscale[1]*gain[1]])+np.min(channel_xy[i]) - gain[1]/2.
        p.append(plt.plot(x,y,'k',clip_on = False,lw = 1))
        t.append(ax.text(np.min(x),np.diff(y)/2. + np.min(y),
                r'{0} $\mu$V'.format(plotscale[1]),
                rotation = 90,va = 'center',ha='right',fontsize=8))
        return p,t
    return p

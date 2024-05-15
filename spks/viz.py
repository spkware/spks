import pylab as plt
import numpy as np
from .utils import discard_nans
from .event_aligned import align_raster_to_event

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

def plot_drift_raster(spike_times,
                      spike_depths,
                      spike_amplitudes,
                      n_spikes_to_plot = 100000,**kwargs):
    '''
    Plot a drift raster: scatter of spike times versus depths, the amplitude is the color
    
    Plots only a subset for speed purposes
    Parameters
    ----------
    spike_times
    spike_depths
    spike_amplitudes
    '''
    idx = np.random.choice(
        np.arange(len(spike_times)),
        np.min([n_spikes_to_plot,len(spike_times)]),
        replace=False)
    idx = idx[np.argsort(np.take(spike_amplitudes,idx,axis=0))]
    

    plt.scatter(spike_times[idx],
                spike_depths[idx], 0.03,
                spike_amplitudes[idx],**kwargs)
    # set the axis
    plt.axis([0,
              np.max(spike_times), # max time
              np.min(spike_depths), # max channel
              np.max(spike_depths)]);# min channel 

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

    rasters = align_raster_to_event(event_times, spike_times, pre_seconds, post_seconds)
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

    rasters = align_raster_to_event(event_times, spike_times, pre_seconds, post_seconds)
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
    #ax.scatter(x, ymin, s=.2, linewidths=.2, marker='|', color=color, **kwargs) #TODO: adaptive s and linewidth kwargs
    ax.scatter(x, ymin, s=5, linewidths=5, marker='|', color=color, **kwargs)
    ax.set_xlabel('Time relative to event (s)')
    #ax.set_ylim((np.min(ymin),np.max(ymax)))
    #ax.set_xlim((np.min(x), np.max(x)))
    #ax.set_ylim(ax.get_ylim()[::-1]) # flip the y axis

def interactive_cluster_waveforms(sp):
    fig = plt.figure()
    ax_waves = fig.add_axes([0.1,0.1,0.8,0.8])
    cluster_positions,principal_channel,active_channels = waveforms_position(sp.cluster_waveforms_mean,sp.channel_positions)
    electrodes = []
    for p in principal_channel:
        electrodes.append(np.where(np.array([np.linalg.norm(c - sp.channel_positions[p]) for c in sp.channel_positions])<100)[0])

    def plot_func(icluster = 20):
        icluster = int(icluster)
        clu_num = sp.unique_clusters[icluster]
        ax_waves.cla()
        ax_waves.plot(sp.channel_positions[:,0],sp.channel_positions[:,1],'kx',markersize = 3,alpha = 0.5)
    
        # wave = sp.get_cluster_waveforms(icluster,n_waveforms=50)
        # plot_footprints(wave[:,:,electrodes[icluster]],sp.channel_positions[electrodes[icluster]],gain=[10,0.05],lw = 0.1,color='k');
        plot_footprints(sp.cluster_waveforms_mean[clu_num][:,electrodes[clu_num]],
                        sp.channel_positions[electrodes[clu_num]],gain=[10,0.05],color='r',lw=1,ax = ax_waves);
        plot_footprints(sp.cluster_waveforms_mean[clu_num][:,electrodes[clu_num]],
                        sp.channel_positions[electrodes[clu_num]],
                        shade_data = sp.cluster_waveforms_std[clu_num][:,electrodes[clu_num]],
                        gain=[10,0.05],color='k',lw=1,ax = ax_waves);
        ax_waves.plot(sp.channel_positions[principal_channel[clu_num],0],
        sp.channel_positions[principal_channel[clu_num],1],'ro',markersize = 5,alpha = 0.5)
        wtime = (1000.*(np.arange(sp.cluster_waveforms_mean.shape[1])-sp.cluster_waveforms_mean.shape[1]/2)/30000).astype(np.float32)

        ax_waves.plot(wtime*10 + sp.channel_positions[principal_channel[clu_num],0],
                    sp.cluster_waveforms_mean[clu_num][:,principal_channel[clu_num]].squeeze()*0.05+sp.channel_positions[principal_channel[clu_num],1],
                                clip_on=False,color = 'b')
        ax_waves.axis('tight')
    plot_func(icluster=20)


    ax_select = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    from matplotlib.widgets import Slider
    sel_slider = Slider(
        ax=ax_select,
        label='cluster',
        valmin=0,
        valmax=len(sp)-1,
        valinit=20,
    )
    sel_slider.on_changed(plot_func)

    def on_press(event):
        if event.key == 'right':
            icluster = np.clip(int(sel_slider.val) + 1,0,len(sp)-1)
            sel_slider.set_val(icluster)
        elif event.key == 'left':
            icluster = np.clip(int(sel_slider.val) - 1,0,len(sp)-1)
            sel_slider.set_val(icluster)

    fig.canvas.mpl_connect('key_press_event', on_press)

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

def plot_footprints(waves, channel_xy, gain = (10,0.05),shade_data = None,shade_color = 'k',shade_alpha=0.3,
                                srate = 30000, plotscale=None, ax = None,**kwargs):
    '''
    Plots multichannel waveforms 
    
    
    Inputs:
    -------
        - waves (nspikes x nsamples x nchannels) or (nsamples x nchannels)
        - channel_xy positions (nchannels x 2)
        - gain (2d tuple)
        - srate (for the scale)
        - plotscale: True to draw the scale
    '''
    import pylab as plt
    
    p = []
    if ax is None:
        ax = plt.gca()
    if len(waves.shape) == 2:
        # plotting mean waveforms (these must be average waveforms)
        waves = waves.copy().reshape(1,*waves.shape)
    wtime = (1000.*(np.arange(waves.shape[1])-waves.shape[1]/2)/srate).astype(np.float32)
    for iw,wv in enumerate(waves):
        for i in range(wv.shape[1]):
            # plot each channel
            p.append(ax.plot(wtime*gain[0] + channel_xy[i,0],
                              wv[:,i]*gain[1]+channel_xy[i,1],
                             clip_on=False,**kwargs))
            if not shade_data is None:
                ax.fill_between(wtime*gain[0] + channel_xy[i,0],
                              (wv[:,i]+shade_data[:,i])*gain[1]+channel_xy[i,1],
                              (wv[:,i]-shade_data[:,i])*gain[1]+channel_xy[i,1],alpha = shade_alpha,
                              edgecolor = None,
                              facecolor=shade_color)
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

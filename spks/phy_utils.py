from .utils import *
from .io import *

def read_phy_data(sortfolder,srate = 30000, bin_file=None, use_kilosort_results=False):
    ''' 
    Read the spike times saved as phy format.
    Does not use template data.
    Computes mean waveforms from the binary file for cluster depth.
    TODO: Add waveform stats and unit depths.
    TODO: Put phy and ks data there if it exists.
    '''
    keys = ['spike_times',
            'spike_clusters']
    tmp = dict() 
    def read_npy(key):
        res = None
        if os.path.isfile(pjoin(sortfolder,key+'.npy')):
            res = np.load(pjoin(sortfolder,key+'.npy'))
        return res
    for k in keys:
        tmp[k] = read_npy(k)
    sp = tmp['spike_times']
    clu = tmp['spike_clusters']
    uclu = np.unique(clu)
   
    if is_phy_curated(sortfolder) and not use_kilosort_results: 
        print('This session has been curated. Labels are from Phy.')
        cgroupsfile = pjoin(sortfolder,'cluster_group.tsv')
    else:
        print('This session has not been curated or user has forced use of kilosort results. Labels are from KS.')
        cgroupsfile = pjoin(sortfolder,'cluster_KSLabel.tsv')
    res = pd.read_csv(cgroupsfile,sep='\t',header = 0)
    
    if bin_file is None:
        spks = [(sp[clu == u]/srate).flatten() for u in res.cluster_id.values]
        res['ts'] = spks

    else:
        print('Reading mean waveforms from the binary file.')
        assert os.path.isfile(bin_file), 'File {0} not found.'.format(bin_file)
        if '.ap.bin' in bin_file:
            meta = read_spikeglx_meta(bin_file.replace('.bin','.meta'))
            chmap = pd.DataFrame(np.vstack([meta['channel_idx'].astype(int),meta['coords'].T]).T,
                                 columns = ['ichan','x','y'])
        else:
            chmap = read_phy_channelmap(sortfolder)

        print('Computing spike times with sRate from meta file. srate argument will be ignored if it was passed!')
        spks = [(sp[clu == u]/meta['sRateHz']).flatten() for u in res.cluster_id.values]
        res['ts'] = spks
        mwaves = par_get_mean_waveforms(bin_file,spks)

        # discard unused channels
        mwaves = mwaves[:,:,np.array(chmap.ichan,dtype=int)]
        res['mean_waveforms'] = [m for m in mwaves]
        # peak to baseline per channel 
        ptb = [mwave[20:50,:].max(axis=0) - mwave[20:50,:].min(axis=0)
               for mwave in mwaves]
        # "active" channels
        activeidx = [np.where(np.abs((p-np.mean(p))/np.std(p))>1)[0]
                     for p in ptb]
        peakchan =  [chmap.ichan.iloc[np.argmax(np.abs(p))] for p in ptb]
        res['peak_channel'] = peakchan
        res['active_channels'] = activeidx
    return res

def phy_replace_params_binary(phyfolder,filename,n_channels_dat = None):
    with open(phyfolder/'params.py','r') as f:
        params = f.read()
    params = params.split('\n')

    for i,p in enumerate(params):
        if p.startswith('dat_path'):
            params[i] = f"dat_path = '{filename}'"
        if not n_channels_dat is None and p.startswith('n_channels_dat'):
            n_channels_dat = int(n_channels_dat)
            params[i] = f"n_channels_dat = {n_channels_dat}"
    
    with open(phyfolder/'params.py','w') as f:
        f.write("\n".join(params))

def read_phy_channelmap(sortfolder):
    ''' 
    
    chmap = read_phy_channelmap(sortfolder)
    
    Reads channelmap from phy (template-gui).

    Needed files: channel_map.npy, channel_positions.npy
    '''
    fname = pjoin(sortfolder,'channel_map.npy')
    assert os.path.isfile(fname),'Could not find {0}'.format(fname)
    chidx = np.load(pjoin(sortfolder,fname))
    fname = pjoin(sortfolder,'channel_positions.npy')
    assert os.path.isfile(fname),'Could not find {0}'.format(fname)
    channel_positions = np.load(pjoin(sortfolder,fname))
    chx = [x[0] for x in channel_positions]
    chy = [x[1] for x in channel_positions]
    return pd.DataFrame(zip(chidx.flatten(),chx,chy),
                        columns = ['ichan','x','y'])

# helper functions to get mean waveforms
vars = {}
def _init_spikeglx_bin(bin_file):
    vars['dat'],vars['meta'] = load_spikeglx_binary(bin_file)
    vars['srate'] = vars['meta']['imSampRate']

def _work_mwaves(ts,chmap = None):
    if chmap is None:
        chmap = np.arange(vars['dat'].shape[1])
    res = get_waveforms(
        vars['dat'],
        chmap,
        (ts*vars['srate']).astype(int),
        nwaves=100,
        npre=30,
        npost=30,
        dofilter=True)
    return np.median(res,axis=0)

def par_get_mean_waveforms(bin_file,ts):
    '''
    mwaves = par_get_mean_waveforms(bin_file,ts)
    Gets the mean waveforms from a binary file given the timestamps.
    Usage:
    
    '''
    from multiprocessing import Pool,cpu_count
    with Pool(processes=cpu_count(),
              initializer=_init_spikeglx_bin,
              initargs=([bin_file])) as pool:
        res = pool.map(_work_mwaves,ts)
    return np.stack(res)



def save_cluster_groups(df,sortfolder):
    '''
    Saves the cluster groups, backs-up previous.
    '''
    olddf = get_cluster_groups(sortfolder,verbose = False)
    folder = None
    sortfolder = os.path.abspath(sortfolder)
    for root, dirs, filenames in os.walk(sortfolder):
        for filename in filenames:
            if not '/.' in root:
                if 'cluster_groups.csv' in filename:
                    folder = root

                    df.to_csv(folder+'/cluster_groups.csv',
                                    index=False,
                                    sep='\t')
                    olddf.to_csv(folder+'/.cluster_groups.bak',
                                index=False,
                                sep='\t')
                    print('Saved cluster_groups in:' + folder)
                    break
    return folder


def get_waveforms(data, datchannels, timestamps, nwaves = 100,
                  srate = 30000, npre = 15, npost=25, random_sample = True,
                  dofilter = True, docar = True):
    '''Gets waveforms sampled randomly from a set of timestamps.'''
    if random_sample:
        spks2extract = np.random.choice(timestamps,
                                        np.clip(nwaves,
                                                1,len(timestamps)),
                                        replace=False)
    else:
        spks2extract = timestamps.flatten()
    indexes = np.arange(-npre,npost,dtype=np.int32)
    waveforms = np.zeros((len(spks2extract),npre+npost,len(datchannels)),
                         dtype=np.float32)
    for i,s in enumerate(np.sort(spks2extract)):
        waveforms[i,:,:] = np.take(data[indexes+s,:].astype(np.float32),datchannels,axis=1)
    if dofilter:
        from scipy import signal
        b,a = signal.butter(3,(500 / (srate / 2.), 5000 / (srate / 2.)),'pass')
        waveforms = signal.filtfilt(b,a,waveforms,axis = 1)
    if docar:
        waveforms = (waveforms.T - np.median(waveforms,axis=2).T).T
    return waveforms

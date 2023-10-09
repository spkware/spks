from .utils import *

def unpackbits_gpu(trace, num_bits = 16, return_binary = False, device = 'cpu'):
    ''' 
    Faster version of unpack_npix_sync that can use the gpu. trace needs to fit in GPU memory.
    Joao Couto - spks 2023
    '''

    if device == 'cuda':
        if not torch.cuda.is_available():
            print('Torch does not have access to the GPU; setting device to "cpu"')
            device = 'cpu'
    # need to include padding also here??
    if isinstance(trace,np.ndarray):
        dtype = trace.dtype
        
        trace = torch.from_numpy(trace.astype('short')).squeeze().to(device)
    # copy to the gpu if not there already
    mask = 2**torch.arange(num_bits).to(trace.device, trace.dtype)
    # converts the bytes to binary
    binary = trace.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
    if return_binary:
        return binary
    # detect the onsets and the offsets
    binary = torch.diff(binary.type(torch.ShortTensor),axis=0).squeeze()

    onsets = {}
    offsets = {}
    for i in range(num_bits):
        ons = torch.nonzero(binary[:,i]>0)  # onsets are positive peaks
        offs = torch.nonzero(binary[:,i]<0) # offsets are negative peaks
        if len(ons):
            if not i in onsets.keys():
                onsets[i] = []
            onsets[i] += [r for r in tensor_to_numpy(ons).flatten()]
            
        if len(offs):
            if not i in offsets.keys():
                offsets[i] = []
            offsets[i] += [r for r in tensor_to_numpy(offs).flatten()]
    return onsets,offsets

def unpack_npix_sync(syncdat,srate=1,output_binary = False):
    '''Unpacks neuropixels phase external input data
events = unpack_npix3a_sync(trigger_data_channel)    
    Inputs:
        syncdat               : trigger data channel to unpack (pass the last channel of the memory mapped file)
        srate (1)             : sampling rate of the data; to convert to time - meta['imSampRate']
        output_binary (False) : outputs the unpacked signal
    Outputs
        events        : dictionary of events. the keys are the channel number, the items the sample times of the events.

    Joao Couto - April 2019

    Usage:
Load and get trigger times in seconds:
    dat,meta = load_spikeglx('test3a.imec.lf.bin')
    srate = meta['imSampRate']
    onsets,offsets = unpack_npix_sync(dat[:,-1],srate);
Plot events:
    plt.figure(figsize = [10,4])
    for ichan,times in onsets.items():
        plt.vlines(times,ichan,ichan+.8,linewidth = 0.5)
    plt.ylabel('Sync channel number'); plt.xlabel('time (s)')
    '''
    dd = unpackbits(syncdat.flatten(),16)
    mult = 1
    if output_binary:
        return dd
    sync_idx_onset = np.where(mult*np.diff(dd,axis = 0)>0)
    sync_idx_offset = np.where(mult*np.diff(dd,axis = 0)<0)
    onsets = {}
    offsets = {}
    for ichan in np.unique(sync_idx_onset[1]):
        onsets[ichan] = sync_idx_onset[0][
            sync_idx_onset[1] == ichan]/srate
    for ichan in np.unique(sync_idx_offset[1]):
        offsets[ichan] = sync_idx_offset[0][
            sync_idx_offset[1] == ichan]/srate
    return onsets,offsets


def unpackbits(x,num_bits = 16):
    '''
    unpacks numbers in bits.
    '''
    xshape = list(x.shape)
    x = x.reshape([-1,1])
    to_and = 2**np.arange(num_bits).reshape([1,num_bits])
    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])


def load_ni_sync_data(sessionpath,apsessions = None,device = 'cpu'):
    '''
    Loads NI and AP sync data for a session
    '''
    if apsessions is None:
        apsessions = [sessionpath]
    niqdsyncpath = list(Path(sessionpath).expanduser().glob('**/*.nidq.bin'))
    if not len(niqdsyncpath):
        print(f'Could not find nidq sync session {sessionpath}')
        return (None,None),(None,None),None
    else:
        niqdsyncpath = niqdsyncpath[0]
    # TODO make this work with other file formats.
    from .spikeglx_utils import load_spikeglx_binary
    sync, meta = load_spikeglx_binary(niqdsyncpath)
    from spks.sync import unpackbits_gpu
    onsets,offsets = unpackbits_gpu(sync[:,-1],device = device)
    return (onsets,offsets),(sync,meta),load_ap_sync_data(apsessions[0])

def load_ap_sync_data(sessionpath):
    '''
    Loads AP sync data for a session from extracted HDF5 data
    '''
    apsyncpaths = list(Path(sessionpath).expanduser().glob('**/*.*.metadata.hdf'))
    if not len(apsyncpaths):
        print(f'Could not find ap sync data {sessionpath}')
        return None
    else:
        # sort the probes
        apsyncdata = [load_dict_from_h5(s) for s in natsorted(apsyncpaths)]
        return apsyncdata
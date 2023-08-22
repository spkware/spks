from .utils import *

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

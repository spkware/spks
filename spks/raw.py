from .utils import *
import torchaudio
from scipy.signal import butter

@torch.no_grad()
def filtfilt_chunk(chunk,a,b,global_car=False, return_gpu = True, device=None, padlen = 150):
    '''
    Filter a chunk of data in the time domain using filter coeffients.
      - chunk are [TIMExCHANNELS] 
      - a and b are the coefficients
    '''
    if device is None:
        device = 'cuda'
    if device == 'cuda': # uses torchaudio
        if not torch.cuda.is_available():
            print('Torch does not have access to the GPU; setting device to "cpu"')
            device = 'cpu'
    # need to include padding also here.
    # make this accept a GPU tensor
    if isinstance(chunk,np.ndarray):
        dtype = chunk.dtype
        T = torch.from_numpy(chunk.astype('float32')).T.to(device)
    else:
        T = chunk.T
    T = torch.nn.functional.pad(T,(padlen,padlen),'reflect') # apply padding for filter
    aa = torch.from_numpy(np.array(a,dtype='float32')).to(device)
    bb = torch.from_numpy(np.array(b,dtype='float32')).to(device)
    X = torchaudio.functional.filtfilt(T,aa,
        bb,clamp=False)
    if global_car:
        X = X-torch.median(X,axis=0).values
    X = X.T[padlen:-padlen,:]
    if 'dtype' in dir():
        # convert to int16 in the gpu, hoping be faster
        if dtype.char in np.typecodes['AllInteger']:
            X = X.type(torch.short)
    if return_gpu:
        return X
    return tensor_to_numpy(X)

def tensor_to_numpy(X):
    '''Converts a tensor to numpy array.''' 
    return X.to('cpu').numpy()

def global_car(chunk,return_gpu = True):
    if isinstance(chunk,np.ndarray):
        dtype = chunk.dtype
        chunk = torch.from_numpy(chunk)
    if chunk.shape[1]<6: # only compute the median if the shape is larger than 6 channels
        X = chunk.T
    else:
        X = chunk.T-torch.median(chunk,axis=1).values
    if return_gpu:
        return X.T
    else:
        return tensor_to_numpy(X.T)

def bandpass_filter(data,sampling_rate, lowpass, highpass, order = 3, device = None, return_gpu = True):
    '''
    bandpass filter
    '''
    sratecoeff = sampling_rate/2.
    b,a = butter(order,[lowpass/sratecoeff, highpass/sratecoeff], btype='bandpass')

    return filtfilt_chunk(data, a , b, device = device, return_gpu = return_gpu)



class RawRecording(object): 
    def __init__(self,files, 
                preprocessing = [lambda x: bandpass_filter(x,30000,300,5000),
                                 lambda x: global_car(x,return_gpu=False)],
                return_preprocessed = True,
                return_voltage = False, **kwargs):
        '''
        Pretend that the recordings are concatenated.
        There is a limit to the chunk size because of how processing is done.
        '''
        # load the files, can be compressed bin or bin
        # get the recording duration by iterating through the files
        self.files = files
        self.current_index = None
        self.nsamples = None
        self.offsets = []
        self.metadata = []
        self.conversion_factor = 1.
        self.preprocessing = preprocessing
        self.return_preprocessed = return_preprocessed
        self.return_voltage = return_voltage
        self._init_parameters()

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, *args):
        # this does not support random temporal indexing
        index  = args[0]
        idx1 = None
        idx2 = None
        if type(index) is tuple: # then look for 2 channels
            if type(index[1]) is slice:
                idx2 = range(*index[1].indices(self.shape[1]))
            elif type(index[1]) in [int,np.int32, np.int64]: # just a frame
                idx2 = [index[1]]
            index = index[0]
        if type(index) is slice:
            idx1 = range(*index.indices(self.shape[0]))#start, index.stop, index.step)
        elif type(index) in [int,np.int32, np.int64]: # just a timesample
            idx1 = [index]
        else: # np.array?
            idx1 = index
        if idx2 is None:
            idx2 = range(self.shape[1])
        # figure out which samples to take
        return self._get_trace(idx1,idx2)
        

    def _get_trace(self,cols,rows,return_preprocessed = None, return_voltage = None):
        if return_preprocessed is None:
            return_preprocessed = self.return_preprocessed
        if return_voltage is None:
            return_voltage = self.return_voltage
        offset = 0
        selidx = np.array(cols,dtype = int)
        buffer = np.zeros((len(selidx),len(rows)),dtype = self.current_pointer.dtype)
        for ifile,(o,f) in enumerate(tt.file_sample_offsets):
            buffidx = np.where((selidx>=o) & (selidx<=f))[0]
            if not len(buffidx):
                continue
            tt._set_current_buffer(ifile)
            fileidx = selidx[buffidx]-o
            tmp = tt.current_pointer[fileidx][:,rows]
            if len(tmp):
                if return_preprocessed:
                    for func in self.preprocessing:
                        tmp = func(tmp)
                buffer[buffidx,:] = tmp
        if return_voltage:
            gains = np.ones(self.shape[1],dtype=np.float32)
            gains[self.channel_info.channel_idx.values.astype(int)] = self.channel_info.conversion_factor.values
            return (buffer.astype(np.float32) * gains[rows])
        return buffer

    def _set_current_buffer(self,ibuffer):
        if not self.current_index == ibuffer:
            self.current_index = ibuffer
            self.current_pointer,meta = load_spikeglx_binary(self.files[self.current_index])

    def _init_parameters(self):
        ''' This function depends on the reader. It should populate the parameters of the object.'''
        for ifile,f in enumerate(self.files):
            if not os.path.exists(f):
                raise(OSError('[RawRecording] - {0} file not found.'.format(f)))
            self.current_pointer,meta = load_spikeglx_binary(f)
            self.offsets.append(self.current_pointer.shape[0])
            self.metadata.append(meta)
            if ifile == 0:
                self.sampling_rate = meta['sRateHz']
                self.channel_info = pd.DataFrame(
                    zip(meta['channel_idx'],meta['coords'],meta['conversion_factor_microV']),
                    columns = ['channel_idx','channel_coord','conversion_factor'])
        self._set_current_buffer(0)
        self.shape = (sum(self.offsets),self.current_pointer.shape[1])
        self.file_sample_offsets = np.vstack([np.hstack([[0],np.cumsum(self.offsets)[:-1]]),
        np.hstack([np.cumsum(self.offsets)])])

    def extract_syncs(self, sync_channel = -1, unpack = True, chunksize = 600000):
        '''Syncs are extracted from the sync channel and converted into onsets and offsets.'''
        from tqdm import tqdm
        for i,f in enumerate(self.files):
            self._set_current_buffer(i)
            dat = self.current_pointer
            chunks = chunk_indices(dat,chunksize=chunksize)
            onsets = {}
            offsets = {}
            for o,f in tqdm(chunks):
                trace = self._get_trace(range(o,f),[-1],return_preprocessed = False,return_voltage = False)
                ons,offs = unpackbits_gpu(trace)
                for o in ons.keys():
                    if not o in onsets.keys():
                        onsets[o] = []
                    onsets[o] += ons[o]
                for o in offs.keys():
                    if not o in offsets.keys():
                        offsets[o] = []
                    offsets[o] += offs[o]

    def to_binary(self,filename, channels = None, as_voltage = False):
        # create a binary file
        pass
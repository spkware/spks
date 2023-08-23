from .utils import *
import torch
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
    T = torch.from_numpy(chunk.astype(np.float32)).T
    T = torch.nn.functional.pad(T.to(device),(padlen,padlen),'reflect') # apply padding for filter
    aa = torch.from_numpy(np.array(a,dtype=np.float32)).to(device)
    bb = torch.from_numpy(np.array(b,dtype=np.float32)).to(device)
    X = torchaudio.functional.filtfilt(T,aa,
        bb,clamp=False)
    if global_car:
        X = X-torch.median(X,axis=0).values
    X = X.T[padlen:-padlen,:]
    if return_gpu:
        return X
    return X.to('cpu').numpy().astype(chunk.dtype)


def bandpass_filter(data,sampling_rate, lowpass, highpass,order = 3, device = None, return_gpu = True):
    '''
    bandpass filter
    '''
    sratecoeff = sampling_rate/2.
    b,a = butter(order,[lowpass/sratecoeff, highpass/sratecoeff], btype='bandpass')

    return filtfilt_chunk(data, a , b, device = device, return_gpu = return_gpu)


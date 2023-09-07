from .utils import *


def waveforms_position(waveforms,channel_positions):
    ''' 
    waveforms [ncluster x nsamples x nchannels]
    '''
    peak_to_peak = (waveforms.max(axis = 1) - waveforms.min(axis = 1))
    # the amplitude of each waveform is the max of the peak difference for all channels
    amplitude = np.abs(peak_to_peak).max(axis=1)
    # compute the center of mass (X,Y) of the templates
    centerofmass = [peak_to_peak*pos for pos in channel_positions.T]
    centerofmass = np.vstack([np.sum(t,axis =1 )/np.sum(peak_to_peak,axis = 1) 
                                        for t in centerofmass]).T
    peak_channels = np.argmax(np.abs(peak_to_peak),axis = 1)
    return centerofmass,peak_channels

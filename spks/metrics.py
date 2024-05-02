from .utils import *

# The goal of these functions are to provide a simple description and hopefully documented implementation of different
#  unit metrics using functions that take the minimum required input and don't depent on data organization.
#  
# There are good resources to learn more about spike sorting metrics:
#  - Hill, D. N. et al. 2011 doi:10.1523/JNEUROSCI.0971-11.2011
#  - Schmitzer-Torbert, N. et al. 2005 doi:10.1016/j.neuroscience.2004.09.066
#  - The Allen institute has a document about unit metrics https://allensdk.readthedocs.io/en/v1.1.1/_static/examples/nb/ecephys_quality_metrics.html
#  - https://github.com/cortex-lab/sortingQuality compiled by Nick Steinmetz, while at the Cortex lab
#  - SpikeInterface documentation - https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics.html
# 
# Check also the waveforms.py file for other metrics that require waveforms.
 
def firing_rate(sp,t_min = None,t_max=None):
    """
    Firing rate between t_min and t_max. 

    It is just the number of spikes divided by the recording duration.
    By default, t_min and t_max are estimated from the time of the 
    first and last spikes, respectively.

    Parameters
    ------------
    sp : np.array
        List of timestamps in the same units as t_min and t_max
        Should be in seconds
    t_min: numeric value, None 
        start time of the experiment, in seconds
    t_max: numeric value, None
        end time of the experiment

    Return
    ------------
    firing_rate: float
        in spikes per second

    Joao Couto - spks 2023
    """
    if len(sp) == 0:
        return 0 # just in case we are not very smart at passing it..
    if t_min is None:
        t_min = sp[0]
    if t_max is None:
        t_max = sp[-1]
    # firing rate is just the number of spikes divided by the time interval
    return float(np.sum((sp>=t_min) & (sp<=t_max)))/(t_max-t_min)

def isi_contamination(ts,refractory_time = 0.0015, censored_time = 0.0002, T = None):
    """
    False positives because of *refractory period violations*.

    In approximation, if gives the false positives rate as a fraction of the unit firing rate.  
    The metric is described in Hill et al. 2011 DOI: https://doi.org/10.1523/JNEUROSCI.0971-11.2011

    This function is adapted as described in Llobet et al. 2022 which reports a 
    generalization of the method in Hill et al 2011 to account also for false positives coming
    from other sources, not only other units.

    We followed equation 3 in Llobet et al. 2022, and assumed :math:`n_v` to be the number of 
    spike pairs that violate the refractory period. We try to keep the names similar in the 
    implementation.

    .. math::
        C=1-\sqrt{1-\frac{n_vT}{N^2t_r}} 

    :math:`n_v` is the number of isi_violations, :math:`N` is the total number of spikes, 
    :math:`T` is the recording/observation duration and :math:`t_r` is the time of the 
    effective refractory period, here *refractory_period* - *censored_time*

    Parameters
    ------------
    sp : np.array
        List of timestamps in the same units as refractory_time and censored_time
    refractory_time: numeric value, 0.0015 
        estimated refractory time to compute violations to the refractory period.
    censored_time: numeric value, 0.0002
        time shorter than the refractory time to discard, for instance because of 
        errors in spike sorting (e.g. double counted spikes)  
    T: int, None
        duration of the recording or observation (to compute the rate). By default it 
        is taken as the time of the first and last spike. That is fair because if the 
        unit was only firing during a fraction of the experiment, the estimate of false
         positives would be lower if one takes the duration of the recording.  

    Return
    ------------
    isi_contamination: float
        Value between 0 and 1 - roughly. Units above 0.1 - 0.2 are likely too contaminated
         to be called single units.     

    Joao Couto - spks 2023
    """

    # sp are sorted since compute the ISI
    N = len(ts)  # number of spikes
    isi = np.diff(ts)  # inter-spike intervals
    n_v = np.sum((isi <= refractory_time) & (isi>censored_time));   # violations between the censored time and the refractory time

    # this is the method/equation described in Lobet et al. 2022 - https://doi.org/10.1101/2022.02.08.479192
    # I assumed n_v is the number of contaminated isis in equation 3
    if T is None:  # recording duration
        T = ts[-1]-ts[0]; # duration of the recording (here the time from the first to the last spike) 
        # in case we pass concatenated stuff but ideally this is an input to the function
    a = 1-n_v*(T-2*N*censored_time)/(N**2*(refractory_time - censored_time))
    if a<0:
        a = np.nan
    isi_contam  = 1 - np.sqrt(a)

    # the Hill method (used in ecephys) underestimates the contamination for contaminations above 0.2
    # violation_time = 2*N*(refractory_time - censored_time)
    # total_rate = N/(ts[-1]-ts[0])
    # violation_rate = n_v/violation_time
    # contam_fraction = violation_rate/total_rate
    return  isi_contam

def isi_contamination_hill(ts,refractory_time = 0.0015, censored_time = 0.0002, T = None):
    """
    False positives because of *refractory period violations*.

    In approximation, if gives the false positives rate as a fraction of the unit firing rate.  
    The metric is described in Hill et al. 2011 DOI: https://doi.org/10.1523/JNEUROSCI.0971-11.2011

    Parameters
    ------------
    sp : np.array
        List of timestamps in the same units as refractory_time and censored_time
    refractory_time: numeric value, 0.0015 
        estimated refractory time to compute violations to the refractory period.
    censored_time: numeric value, 0
        time shorter than the refractory time to discard, for instance because of 
        errors in spike sorting (e.g. double counted spikes)  
    T: int, None
        duration of the recording or observation (to compute the rate). By default it 
        is taken as the time of the first and last spike. That is fair because if the 
        unit was only firing during a fraction of the experiment, the estimate of false
         positives would be lower if one takes the duration of the recording.  

    Return
    ------------
    isi_contamination: float
        Value between 0 and 1 - roughly. Units above 0.1 - 0.2 are likely too contaminated
         to be called single units.     

    Joao Couto - spks 2023
    """

    # sp are sorted since compute the ISI
    N = len(ts)  # number of spikes
    isi = np.diff(ts)  # inter-spike intervals
    n_v = np.sum((isi <= refractory_time) & (isi>censored_time));   # violations between the censored time and the refractory time

    if T is None:  # recording duration
        T = ts[-1]-ts[0]; # duration of the recording (here the time from the first to the last spike)
    # the Hill method (used in ecephys) underestimates the contamination for contaminations above 0.2
    violation_time = 2*N*(refractory_time - censored_time)
    total_rate = N/(T)
    isi_contam = n_v/violation_time
    # contam_fraction = violation_rate/total_rate
    return  isi_contam

def presence_ratio(sp,t_min, t_max, min_spikes = 1,nbins = 100):
    """
    Computes the *presence ratio* - the fraction of time a unit fires.
    
    It can be used to identify units that fire only in a fraction of the experiment 
but without caring for which fraction. Units may have low presence ratio because of 
multiple reasons:
        - some units are very selective and fire only to specific stimuli 
        - if there is tissue drift in relation to the probe, spike sorting may bundle cells
into differnt units or stop detecting a cell because it is out of reach

    This function makes a histogram with *nbins* between *t_min* and *t_max* and 
and counts how many bins have more than *min_spikes*.

    Parameters
    ------------
    sp : np.array
        List of timestamps in the same units as t_min and t_max
    t_min: numeric value
        Start time of the recording
    t_max: numeric value
        End time of the recording
    min_spikes: int
        number of spikes in a bin to consider the unit present in that bin
    nbins: int
        number of bins

    Return
    ------------
    presence_ratio: float
        Fraction of the total time the cell was firing.
        Value between 0 and 1. 1 is present in the whole experiment    

    Joao Couto - spks 2023
    """

    counts,_ = np.histogram(sp, np.linspace(t_min, t_max, nbins+1)) # add 1 so the number of bins is actually nbins

    return np.sum(counts > min_spikes)/nbins


def amplitude_cutoff(amplitudes, num_histogram_bins = 500, histogram_smoothing_value = 3):

    """ This is from the Allen Institute - ecephys implementation.
    Calculates approximate fraction of spikes missing from a distribution of amplitudes
    
    Assumes the amplitude histogram is symmetric (not valid in the presence of drift)
    
    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705

    Parameters:
    ------
    amplitudes : numpy.ndarray
    Array of amplitudes (don't need to be in physical units)

    Output:
    -------
    fraction_missing : float
    Fraction of missing spikes (0-0.5)
    If more than 50% of spikes are missing, an accurate estimate isn't possible
    """


    h,b = np.histogram(amplitudes, num_histogram_bins, density=True)

    pdf = gaussian_filter(h,histogram_smoothing_value)
    support = b[:-1]

    peak_index = np.argmax(pdf)
    G = np.argmin(np.abs(pdf[peak_index:] - pdf[0])) + peak_index

    bin_size = np.mean(np.diff(support))
    fraction_missing = np.sum(pdf[G:])*bin_size

    fraction_missing = np.min([fraction_missing, 0.5])

    return fraction_missing

def depth_stability(sptimes,
                    spdepths,
                    tmin = 0,
                    tmax = None,
                    min_spikes = 5,
                    bin_size = 60, # seconds
                   ):
    '''
    Calculate the depth stability of a unit from the spike times and depths.
    This is also commonly called "drift"; this is inspired in the ecephys metrics. 
    
    The max_depth_range is obtained by subtracting the 2 depth extrema 
(of the median  binned depths over "binsize").
    The mean depth fluctuation is the average fluctuation between subsequent 
time periods (the mean so we can compare units recorded for differnet time durations).

    Joao Couto - spks 2024
    '''
    if tmax is None:
        tmax = np.max(sptimes)
    edges = np.arange(0,tmax,bin_size)
    
    bins = np.digitize(sptimes,edges)
    med_depths = np.zeros(len(edges))
    med_depths[:] = np.nan
    for i in range(len(edges)):
        t = spdepths[bins == i+1]
        if len(t)>min_spikes:
            med_depths[i] = np.median(t)
    max_depth_range = np.nanmax(med_depths) - np.nanmin(med_depths)
    mean_depth_fluctuation = np.nanmean(np.abs(np.diff(med_depths)))
    return max_depth_range,mean_depth_fluctuation

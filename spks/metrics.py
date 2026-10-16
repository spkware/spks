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

def isi_violations(ts,refractory_time = 0.0015, censored_time = 0.0002):
    """
    Number of *refractory period violations*.

    Counts the inter-spike intervals longer than *censored_time* and shorter than or
    equal to *refractory_time*. This is the :math:`n_v` used by isi_contamination and
    isi_contamination_hill.

    Parameters
    ------------
    ts : np.array
        Sorted timestamps in the same units as refractory_time and censored_time
    refractory_time: numeric value, 0.0015
        estimated refractory time to compute violations to the refractory period.
    censored_time: numeric value, 0.0002
        time shorter than the refractory time to discard, for instance because of
        errors in spike sorting (e.g. double counted spikes)

    Return
    ------------
    n_violations: int
        Number of inter-spike intervals that violate the refractory period.

    Joao Couto - spks 2026
    """
    isi = np.diff(ts)  # inter-spike intervals
    return int(np.sum((isi <= refractory_time) & (isi > censored_time)))

def isi_contamination(ts,refractory_time = 0.0015, censored_time = 0.0002, T = None):
    r"""
    False positives because of *refractory period violations*.

    This function is adapted as described in Llobet et al. 2022 (DOI 10.1101/2022.02.08.479192) 
    which reports a generalization of the method in Hill et al 2011 to account also for false 
    positives coming from other sources, not only other units.

    We followed equation 3 in Llobet et al. 2022, and assumed :math:`n_v` to be the number of 
    spike pairs that violate the refractory period. We try to keep the names similar in the 
    implementation.

    .. math::
        C=1-\sqrt{1-\frac{n_vT}{N^2t_r}} 

    :math:`n_v` is the number of isi_violations, :math:`N` is the total number of spikes, 
    :math:`T` is the recording/observation duration and :math:`t_r` is the time of the 
    effective refractory period, here *refractory_period* - *censored_time*

See also Hill et al. 2011 DOI: https://doi.org/10.1523/JNEUROSCI.0971-11.2011

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
        NaN when the term under the square root is negative; the equation has no
        solution there, which happens for heavily contaminated units. isi_violations
        and isi_contamination_hill still give a value for those.

    Joao Couto - spks 2023
    """

    # sp are sorted since compute the ISI
    N = len(ts)  # number of spikes
    n_v = isi_violations(ts, refractory_time, censored_time)  # violations between the censored time and the refractory time

    # this is the method/equation described in Llobet et al. 2022 - https://doi.org/10.1101/2022.02.08.479192
    # I assumed n_v is the number of contaminated isis in equation 3
    if T is None:  # recording duration
        T = ts[-1]-ts[0]; # duration of the recording (here the time from the first to the last spike) 
        # in case we pass concatenated stuff but ideally this is an input to the function
    a = 1-n_v*(T-2*N*censored_time)/(N**2*(refractory_time - censored_time))
    if a < 0:
        return np.nan
    
    isi_contam  = 1 - np.sqrt(a)

    return  isi_contam

def isi_contamination_hill(ts,refractory_time = 0.0015, censored_time = 0.0002, T = None):
    """
    False positives because of *refractory period violations*.

    Gives the rate of false positives as a fraction of the unit firing rate. Hill et al. 2011
    (DOI: https://doi.org/10.1523/JNEUROSCI.0971-11.2011) relate the violations to the
    fraction of false positives :math:`f_p` with

    .. math::
        n_v = \\frac{2 t_r N^2 f_p (1 - f_p)}{T}

    This function returns the first order approximation (as in ecephys and SpikeInterface),
    which drops the :math:`(1 - f_p)` term:

    .. math::
        f_p \\approx \\frac{n_v T}{2 N^2 t_r}

    with :math:`t_r` the refractory time minus the censored time.

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
        Fraction of false positives, dimensionless. Units above 0.1 - 0.2 are likely too
        contaminated to be called single units. The approximation underestimates the
        contamination above about 0.2 and can exceed 1 for heavily contaminated units.

    Joao Couto - spks 2023
    """

    # sp are sorted since compute the ISI
    N = len(ts)  # number of spikes
    n_v = isi_violations(ts, refractory_time, censored_time)  # violations between the censored time and the refractory time

    if T is None:  # recording duration
        T = ts[-1]-ts[0]; # duration of the recording (here the time from the first to the last spike)
    # the Hill method (as in ecephys) underestimates the contamination for contaminations above 0.2
    violation_time = 2*N*(refractory_time - censored_time)
    total_rate = N/(T)
    violation_rate = n_v/violation_time
    return  violation_rate/total_rate

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
                    min_spikes = 5, # min number of spikes to compute median waveforms 
                    bin_size = 60, # bin size to compute median waveforms seconds
                    frac_compare = 0.2 # fraction of the experiment to compare start and end drift
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
        if len(t) > min_spikes:
            med_depths[i] = np.median(t)
    start_to_end_drift = np.nanmedian(med_depths[edges[::-1]<tmax*frac_compare]) - np.nanmedian(med_depths[edges[:]<tmax*frac_compare])

    med_depths = med_depths[np.isfinite(med_depths)]
    if len(med_depths) < 3: # in case there aren't enough spikes
        return np.nan, np.nan, np.nan
    max_depth_range = np.nanmax(med_depths) - np.nanmin(med_depths)
    mean_depth_fluctuation = np.nanmean(np.abs(np.diff(med_depths)))
    return max_depth_range,mean_depth_fluctuation,start_to_end_drift

def firing_range(sp, t_min, t_max, bin_size = 10, percentiles = (5, 95)):
    '''
    Range of the firing rate over the recording.

    Bins the spikes in *bin_size* windows between *t_min* and *t_max* and returns the 
difference between the *percentiles* of the binned firing rate. A unit that drifts out 
of reach or stops firing for part of the experiment has a range that is large compared 
to its firing rate. Inspired by the firing_range metric of SpikeInterface.

    Parameters
    ------------
    sp : np.array
        List of timestamps in seconds
    t_min: numeric value
        Start time of the recording
    t_max: numeric value
        End time of the recording
    bin_size: numeric value, 10
        size of the bins in seconds; a last bin shorter than *bin_size* is left out
    percentiles: tuple, (5, 95)
        lower and upper percentiles of the binned firing rate

    Return
    ------------
    firing_range: float
        in spikes per second; NaN when the recording holds fewer than 2 bins

    Joao Couto - spks 2026
    '''
    nbins = int((t_max - t_min)//bin_size)
    if nbins < 2:
        return np.nan
    edges = t_min + bin_size*np.arange(nbins + 1)
    rates = np.histogram(sp, edges)[0]/bin_size
    low, high = np.percentile(rates, percentiles)
    return float(high - low)

def amplitude_drift(sptimes, amplitudes, t_min, t_max, frac_compare = 0.2, min_spikes = 5):
    '''
    Change of the spike amplitude from the start to the end of the recording.

    Ratio between the median amplitude of the spikes in the last *frac_compare* of the 
recording and that of the spikes in the first *frac_compare*. A stable unit gives 1, 
a unit that shrinks (e.g. drifts away from the electrodes) less than 1 and one that 
grows more than 1.

    Parameters
    ------------
    sptimes : np.array
        List of timestamps in the same units as t_min and t_max
    amplitudes : np.array
        Amplitude of each spike (e.g. the template amplitudes)
    t_min: numeric value
        Start time of the recording
    t_max: numeric value
        End time of the recording
    frac_compare: float, 0.2
        fraction of the recording compared at the start and at the end
    min_spikes: int, 5
        minimum number of spikes in each window

    Return
    ------------
    amplitude_drift: float
        median end amplitude over median start amplitude; NaN when either window has 
        fewer than *min_spikes* spikes

    Joao Couto - spks 2026
    '''
    sptimes = np.asarray(sptimes).flatten()
    amplitudes = np.asarray(amplitudes).flatten()
    window = (t_max - t_min)*frac_compare
    start = amplitudes[sptimes < (t_min + window)]
    end = amplitudes[sptimes > (t_max - window)]
    if len(start) < min_spikes or len(end) < min_spikes:
        return np.nan
    start = np.median(start)
    if start == 0:
        return np.nan
    return float(np.median(end)/start)

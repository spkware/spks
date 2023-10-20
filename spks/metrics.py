from .utils import *
#TODO: Add the docs to these functions...


def firing_rate(sp,t_min = None,t_max=None):
    if len(sp) == 0:
        return 0 # just in case we are not very smart at passing it..
    if t_min is None:
        t_min = sp[0]
    if t_max is None:
        t_max = sp[-1]
    # firing rate is just the number of spikes divided by the time interval
    return float(np.sum((sp>=t_min) & (sp<=t_max)))/(t_max-t_min)

def isi_contamination(ts,refractory_time = 0.0015, censored_time = 0.0005, T = None):
    # sp are sorted since compute the ISI
    N = len(ts)  # number of spikes
    isi = np.diff(ts)  # inter-spike intervals
    n_v = np.sum((isi <= refractory_time) & (isi>censored_time));   # violations between the censored time and the refractory time

    # this is the equation described in Lobet et al. 2022 - https://doi.org/10.1101/2022.02.08.479192
    # I assumed n_v is the number of contaminated isis
    if T is None:  # recording duration
        T = ts[-1]-ts[0]; # duration of the recording (here the time from the first to the last spike) 
        # in case we pass concatenated stuff but ideally this is an input to the function
    isi_contam  = 1 - np.sqrt(1-n_v*(T-2*N*censored_time)/(N**2*(refractory_time - censored_time)))

    # the Hill method (used in ecephys)
    # violation_time = 2*N*(refractory_time - censored_time)
    # total_rate = N/(ts[-1]-ts[0])
    
    # violation_rate = n_v/violation_time
    # contam_fraction = violation_rate/total_rate
    return  isi_contam


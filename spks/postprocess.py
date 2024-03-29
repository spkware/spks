from .utils import *
from .waveforms import *

def get_overlapping_spikes_indices(spike_times,
                                   spike_clusters,
                                   mwaves,channel_positions,
                                   nsamples_threshold = 5,
                                   distance_radius = 50):
    from tqdm import tqdm
    '''

    doublecounted = get_overlapping_spikes_indices(spike_times,spike_clusters,mwaves,channel_positions)

        - spike_times nspikesx1 array
        - spike_clusters nspikesx1 array - which spike corresponds to which cluster
        - mwaves nclustersxntimesamplesxnchannels
        - channel_positions nchannelsx2 (x and y channel positions)

        - doublecounted - indices of double counted spikes

        Use spike_times = np.delete(spike_times, doublecounted) to remove double counted spikes.


    Double counted spikes can occur in 2 situations:
      1. in the same unit during template matching (these can be found easily by checking the diff of the spike indices)
      2. between 2 units, usually when one template has a low amplitude (https://github.com/MouseLand/Kilosort/issues/29). 
    5 samples (0.16ms) is a reasonable number to consider a double counted spike as in the ecephys package. We will use the same by default. 
    In this package, we will remove double counted spikes from each unit first, then we will remove only the duplicate spikes from the cluster with the smallest waveform
    this function will only return the indices, you can delete those indices using np.delete

    It requires the mean waveforms to compute the clusters that are close to each other. It also requires the channel_positions for that reason.
    The mean waveforms are also used to compute the amplitude. Duplicated spikes can occur when the best match is a combine
    The templates are the closest to the mean waveforms and can be used if no merges were done in phy.

    See (https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/kilosort_postprocessing)

    Joao Couto - spks, 2023
   ''' 

    ts = np.copy(spike_times).flatten()
    clus = np.copy(spike_clusters).flatten()

    assert len(ts)==len(clus), "[get_overlapping_spikes_indices] - spike_clusters and spike_times don't have the same number of spikes..."
    org_idx = np.arange(len(ts),dtype=np.uint64)
    # this needs to be done cluster by cluster
    unique_clusters = np.unique(clus)

    indices = []
    for aclu in unique_clusters:
        s = ts[clus==aclu]
        idx = np.where(np.diff(s)<nsamples_threshold)[0]
        if len(idx):
            indices.append(org_idx[clus==aclu][idx])
    to_delete = []
    if len(indices):
        to_delete = np.hstack(indices)
     
    # apply so we can look at cross-unit duplicates.
    print('get_overlapping_spikes_indices] - found {0} "double counted" of {1} within unit spikes'.format(len(to_delete)))
    org_idx = np.delete(org_idx,to_delete)
    ts = np.delete(ts,to_delete)
    clus = np.delete(clus,to_delete)

    # need the waveform amplitude and the location to know if to keep or discard
    position, _ = waveforms_position(mwaves,channel_positions)
    peak_to_peak = (mwaves.max(axis = 1) - mwaves.min(axis = 1)).max(axis=1)
    duplicated = []
    
    # only attempt to search for duplicates for units that are within radius and do it only once
    indices = []
    for ia,aclu in tqdm(enumerate(unique_clusters), desc='Finding duplicate spikes across units',total=len(unique_clusters)):
        for ib,bclu in enumerate(unique_clusters):
            if (ia>ib) and (np.linalg.norm(position[ia] - position[ib]) < distance_radius):
                # for an alternative see ecephys, this keeps the spikes in the largest unit.
                s = np.sort(np.hstack([ts[clus==aclu],ts[clus==bclu]])) # the spike times for both units combined
                duplicated = np.where(np.diff(s)<nsamples_threshold)[0]
                duplicated_spikes = np.hstack([s[duplicated],s[duplicated+1]])
                if len(duplicated): # Then there are spikes to remove (the spikes are in duplicated_spikes)
                    # go to the smallest unit and get the index for those spikes
                    if np.argmin(peak_to_peak[[ia,ib]]) == 0:
                        # find the indices of the spikes in cluster a
                        indices.append(org_idx[clus == aclu][np.isin(ts[clus==aclu], duplicated_spikes)])
                    else:
                        # find the indices of the spikes in cluster b
                        indices.append(org_idx[clus == bclu][np.isin(ts[clus==bclu], duplicated_spikes)])
    # only delete once (unique) and export also the ones for inside each unit
    to_delete = np.unique(np.hstack([to_delete]+indices)).flatten().astype(np.uint64) # this should probably be unsigned 64
    print("[get_overlapping_spikes_indices] - found {0} 'double counted' of {1} spikes".format(len(to_delete),len(spike_times)))
    return to_delete


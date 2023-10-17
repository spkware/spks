from .utils import *
from .waveforms import *

class Clusters():
    def __init__(self,folder = None,
                spike_times = None, 
                spike_clusters = None,
                channel_positions = None, 
                channel_map = None,
                sampling_rate = None,
                channel_gains = None,
                get_waveforms = True,
                compute_raw_templates=True):#, remove_duplicate_spikes = False):
        '''
        Object to access the spike sorting results like an array

        Inputs
        ------
            - folder (A folder with phy results)
            - (optional) spike_times
            - (optional) spike_clusters
            - (optional) channel_positions
            - get_waveforms

        Joao Couto - spks 2023
        '''
        if type(folder) in [str]:
            folder = Path(folder)
        self.folder = folder
        # load spiketimes
        self.spike_times = self._load_required('spike_times.npy',spike_times)
        # load each spike cluster number
        self.spike_clusters = self._load_required('spike_clusters.npy',spike_clusters)
        unique_clusters = np.sort(np.unique(self.spike_clusters)).astype(int)
        self.cluster_info = pd.DataFrame(dict(cluster_id = unique_clusters))
        # load the channel locations
        self.channel_positions =  self._load_optional('channel_positions.npy',channel_positions)
        self.channel_map =  self._load_optional('channel_map.npy',channel_map)
        self.channel_gains = None

        # Load the amplitudes used to fit the template
        self.spike_template_amplitudes = self._load_optional('amplitudes.npy')
        # load spike templates (which template was fitted) for each spike
        self.spike_templates = self._load_optional('spike_templates.npy')
        # load the templates used to extract the spikes
        self.templates =  self._load_optional('templates.npy')
        # load the whitening matrix (to correct for the data having been whitened)
        self.whitening_matrix = self._load_optional('whitening_mat_inv.npy')
        if not self.whitening_matrix is None:
            self.whitening_matrix = self.whitening_matrix.T
        self.cluster_groups = self._load_optional('cluster_group.tsv')
        
        self.cluster_waveforms_mean = None
        self.cluster_waveforms_std = None
        self.cluster_waveforms = None
        self.sampling_rate = None


        if get_waveforms:
            self.load_waveforms()
        # compute the raw templates and the position of each cluster based on the template position
        if compute_raw_templates:
            self._compute_template_amplitudes()

        if self.sampling_rate is None:
            self.sampling_rate = 30000.
        self.compute_statistics(npre = 30, srate = self.sampling_rate)  # computes the statistics
        
        self.update_cluster_info()
        #if remove_duplicate_spikes:
        #    self.remove_duplicate_spikes()
    def update_cluster_info(self): 
        self.__dict__.update(dict(zip(list(self.cluster_info), self.cluster_info.to_numpy().T)))

    def get_cluster(self,cluster_id):
        self.spike_times[self.spike_clusters == cluster_id]

    def __getitem__(self, index):
        ''' returns the spiketimes for a set of clusters'''
        if type(index) in [int,np.int64,np.int32]:
            index = [index]
        if type(index) in [slice]:
            index = np.arange(*index.indices(len(self)))
        sp = []
        for iclu in self.cluster_id[index]:
            sp.append(self.spike_times[self.spike_clusters == iclu])
        if len(sp) == 1:
            return sp[0]
        else:
            return sp

    def compute_statistics(self,npre,srate):
        '''
        Gets waveform and unit metrics 
        This will compute all metrics possible (some depend on having loaded waveforms). 
        The metrics are:
            - TODO
        '''
        from .waveforms import compute_waveform_metrics
        # computes the metrics and appends to cluster_info
        clumetrics = []
        for iclu,waveforms,cluster_channel in tqdm(zip(self.cluster_info.cluster_id.values,self.cluster_waveforms_mean,self.cluster_channel),
        desc = 'Computing cluster waveform stats'):
            wave = waveforms[:,cluster_channel]
            clumetrics.append(dict(cluster_id = iclu,
                              **compute_waveform_metrics(wave,npre=npre,srate=srate)))
        clumetrics = pd.DataFrame(clumetrics)
        activeidx = estimate_active_channels(self.cluster_waveforms_mean)
        nactive_channels = np.array([len(a) for a in activeidx])
        clumetrics['n_active_channels'] = nactive_channels
        clumetrics['active_channels'] = activeidx
        self.cluster_info = pd.merge(self.cluster_info,clumetrics)


    def __len__(self):
        return len(self.cluster_info)

    def __iter__(self):
        for iclu in self.cluster_id:
            yield self.spike_times[self.spike_clusters == iclu]


    def remove_duplicate_spikes(self,overwrite_phy = False):
        from .postprocess import get_overlapping_spikes_indices
        doubled = get_overlapping_spikes_indices(self.spike_times,self.spike_clusters, self.templates_raw, self.channel_positions)
        if not len(doubled):
            return
        self.spike_times = np.delete(self.spike_times,doubled)
        self.spike_clusters = np.delete(self.spike_clusters,doubled)
        
        if not self.spike_amplitudes is None:
            self.spike_amplitudes = np.delete(self.spike_amplitudes,doubled)
        if not self.spike_positions is None:
            self.spike_positions = np.delete(self.spike_positions,doubled)
        if not self.spike_templates is None:
            self.spike_templates = np.delete(self.spike_templates,doubled)
        if not self.spike_template_amplitudes is None:
            self.spike_template_amplitudes = np.delete(self.spike_template_amplitudes,doubled)
        if overwrite_phy:
            self.export_phy(self.folder)

    def export_phy(self,folder):
        if type(folder) is str:
            folder = Path(folder)
        np.save(folder/'spike_times.npy',self.spike_times)
        np.save(folder/'spike_clusters.npy',self.spike_clusters)
        if not self.spike_template_amplitudes is None:
            np.save(folder/'amplitudes.npy', self.spike_template_amplitudes)
        if not self.spike_templates is None:
            np.save(folder/'spike_templates.npy', self.spike_templates)
    
    def _compute_template_amplitudes(self):
        self.templates_raw = None
        self.templates_amplitude = None
        self.templates_position = None
        self.spike_amplitudes = None
        self.spike_positions = None
        if (not self.templates is None and 
            not self.whitening_matrix is None and 
            not self.channel_positions is None):
            # the raw templates are the dot product of the templates by the whitening matrix
            self.templates_raw = np.dot(self.templates,self.whitening_matrix)
            # compute the peak to peak of each template
            templates_peak_to_peak = (self.templates_raw.max(axis = 1) - self.templates_raw.min(axis = 1))
            # the amplitude of each template is the max of the peak difference for all channels
            self.templates_amplitude = templates_peak_to_peak.max(axis=1)
            # compute the center of mass (X,Y) of the templates
            from .waveforms import waveforms_position
            self.template_position,self.template_channel = waveforms_position(self.templates_raw,self.channel_positions)
            # get the spike positions and amplitudes from the average templates
            self.spike_amplitudes = self.templates_amplitude[self.spike_templates]*self.spike_template_amplitudes
            self.spike_positions = self.template_position[self.spike_templates,:].squeeze()

    def get_cluster_waveforms(self,cluster_id,n_waveforms = 50):
        if not hasattr(self,'cluster_waveforms'):
            self.load_waveforms()
        if str(cluster_id) in self.cluster_waveforms.keys():
            shpe = self.cluster_waveforms[str(cluster_id)].shape[0]
            idx = np.sort(np.random.choice(range(shpe),size=min(n_waveforms,shpe),replace=False))
            return self.cluster_waveforms[str(cluster_id)][idx]

    def load_waveforms(self):
        '''Loads waveform saved in a cluster_waveforms.hdf file.'''
        if not self.folder is None:
            if (self.folder/'cluster_waveforms.hdf').exists():
                self.cluster_waveforms = h5.File((self.folder/'cluster_waveforms.hdf'),'r')
                with Pool(12) as pool:
                    result = [r for r in tqdm(pool.imap(partial(_mean_std_from_cluster_waveforms,
                                            folder = self.folder),self.cluster_info.cluster_id.values),
                                    desc='[Clusters] Computing mean waveforms', total = len(self))]
                    self.cluster_waveforms_mean = np.stack([r[0] for r in result])
                    self.cluster_waveforms_std = np.stack([r[1] for r in result])
                from .waveforms import waveforms_position
                self.cluster_position, self.cluster_channel = waveforms_position(self.cluster_waveforms_mean, self.channel_positions)
                self.cluster_info['depth'] = self.cluster_position[:,1]
                self.cluster_info['electrode'] = self.cluster_channel

        if not hasattr(self,'cluster_waveforms'):
            print()
            raise(OSError('[Clusters] - Waveforms file [cluster_waveforms.hdf] not in folder'))

    def _load_required(self,file,var = None):
        '''
        Loads a required variable, if the file path does not exist returns an error, 
        unless var is specified, in that case it returns var
        '''
        if var is None:
            path = self.folder / file
            assert path.exists(), f'[Clusters] - {path} doesnt exist'
            return np.load(path)
        return var

    def _load_optional(self,file,var = None):
        '''Loads an optional variable, if the file path does not exist returns None, 
        unless var is specified, in that case it returns var'''
        if var is None:
            path = self.folder / file
            if path.exists():
                if path.suffix == '.npy':
                    return np.load(path)
                elif path.suffix == '.tsv':
                    return pd.read_csv(path,sep = '\t')
            return None
        else:
            return var


########################################################
################HELPER FUNCTIONS########################
########################################################


def _mean_std_from_cluster_waveforms(icluster,folder):
    '''Computes the average and std of te waveforms for a specific cluster in a hdf file'''
    import h5py as h5
    with  h5.File(folder/'cluster_waveforms.hdf','r') as waveforms_file:
        mwave = np.median(waveforms_file[str(icluster)][()],axis = 0) 
        stdwave = np.std(waveforms_file[str(icluster)][()],axis = 0)#/waveforms_file[str(icluster)].shape[0]
    return np.stack([mwave,stdwave])

    
    
    
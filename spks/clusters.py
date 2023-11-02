from .utils import *
from .waveforms import *

class Clusters():
    def __init__(self,folder = None,
                spike_times = None, 
                spike_clusters = None,
                spike_template_amplitudes = None,
                spike_templates = None,
                spike_pc_features = None,
                template_pc_features_ind = None,
                channel_positions = None, 
                channel_map = None,
                sampling_rate = None,
                channel_gains = None,
                get_waveforms = True,
                get_metrics = True,
                name = 'Clusters',
                load_template_features = True,
                compute_raw_templates=True):
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
        self.name = name
        if type(folder) in [str]:
            folder = Path(folder)
        self.folder = folder
        # load spiketimes
        self.spike_times = self._load_required('spike_times.npy',spike_times)
        # load each spike cluster number
        self.spike_clusters = self._load_required('spike_clusters.npy',spike_clusters)

        self._set_auxiliary()  # set other variables that used to speed up things internally

        unique_clusters = np.sort(np.unique(self.spike_clusters)).astype(int)
        self.cluster_info = pd.DataFrame(dict(cluster_id = unique_clusters))
        # load the channel locations
        self.channel_positions =  self._load_optional('channel_positions.npy',channel_positions)
        self.channel_map =  self._load_optional('channel_map.npy',channel_map)
        self.channel_gains = channel_gains

        if load_template_features:  # this can take a while, disable if you don't need to estimate amplitudes or so
            # Load the amplitudes used to fit the template
            self.spike_template_amplitudes = self._load_optional('amplitudes.npy',spike_template_amplitudes)
            # load spike templates (which template was fitted) for each spike
            self.spike_templates = self._load_optional('spike_templates.npy',spike_templates)
            # load the principle component features for each spike nspikes x featuresperchannel x nfeatures            
            self.spike_pc_features = self._load_optional('pc_features.npy',spike_pc_features)
            # template index for each PC feature
            self.template_pc_features_ind = self._load_optional('pc_feature_ind.npy',template_pc_features_ind)
            # load the templates used to extract the spikes
            self.templates =  self._load_optional('templates.npy')
            # load the whitening matrix (to correct for the templates having been whitened)
            self.whitening_matrix = self._load_optional('whitening_mat_inv.npy')
            if not self.whitening_matrix is None:
                self.whitening_matrix = self.whitening_matrix.T
        self.cluster_groups = self._load_optional('cluster_group.tsv')
        
        self.cluster_waveforms_mean = None
        self.cluster_waveforms_std = None        
        self.cluster_waveforms = None
        self.sampling_rate = sampling_rate

        # is there a sorting metadata file on this folder?
        metadatafile = list(Path(self.folder).glob('filtered_recording*metadata.hdf'))
        self.metadata = None
        if len(metadatafile):
            self.metadata = load_dict_from_h5(metadatafile[0])
            if self.sampling_rate is None:
                self.sampling_rate = float(self.metadata['sampling_rate'])
            if self.channel_gains is None:
                self.channel_gains = self.metadata['channel_conversion_factor'].flatten()

        # compute the raw templates and the position of each cluster based on the template position
        if compute_raw_templates:
            self._compute_template_amplitudes_and_depths()
        # load waveforms
        if get_waveforms:
            self.load_waveforms()
        # load the sampling rate?
        if self.sampling_rate is None:
            self.sampling_rate = 30000.
        # get or compute metrics
        if get_metrics:
            self.compute_statistics(npre = 30, srate = self.sampling_rate)  # computes the statistics

        self.update_cluster_info()
        #if remove_duplicate_spikes:
        #    self.remove_duplicate_spikes()
    def extract_waveforms(self,data, chmap = None, max_n_spikes = 500, npre = 45, npost = 45, chunksize = 10, save_waveforms = True):
        
        if chmap is None:
            chmap = np.arange(data.shape[1],dtype = int)
        from .waveforms import extract_waveform_set
        mwaves = extract_waveform_set(
            spike_times = self,
            data = data,
            chmap = chmap,
            npre = npre,
            npost = npost,
            max_n_spikes = max_n_spikes, # max number of spikes to extract for the average waveforms
            chunksize = chunksize)
        
        waveforms = {}
        for iclu,w in zip(sp.cluster_id,mwaves):
            waveforms[int(iclu)] = w
        if save_waveforms:
            save_dict_to_h5(resultsfolder/'cluster_waveforms.hdf',waveforms)
        self.load_waveforms()
        return waveforms

    def __del__(self):
        if hasattr(self,'cluster_waveforms'):
            # close the waveforms if the hdf5 is open
            del self.cluster_waveforms

    def _set_auxiliary(self):
        # uses torch indexing which is faster.
        device = 'cpu' # no point in overloading the gpu
        self._spike_clusters_t = torch.from_numpy(self.spike_clusters.astype('int32')).squeeze().to(device) 

    def update_cluster_info(self): 
        self.__dict__.update(dict(zip(list(self.cluster_info), self.cluster_info.to_numpy().T)))

    def get_cluster(self,cluster_id):
        # return self.spike_times[self.spike_clusters == cluster_id]  # roughly twice slower than below. 
        return np.take(self.spike_times,torch.nonzero(self._spike_clusters_t == cluster_id).flatten())

    def __getitem__(self, index):
        ''' returns the spiketimes for a set of clusters'''
        if type(index) in [int,np.int64,np.int32]:
            index = [index]
        if type(index) in [slice]:
            index = np.arange(*index.indices(len(self)))
        sp = []
        for iclu in self.cluster_info.cluster_id.values[index]:
            sp.append(self.get_cluster(iclu))  
        if len(sp) == 1:
            return sp[0]
        else:
            return sp

    def compute_statistics(self,npre,srate):
        '''
        Gets waveform and unit metrics 
        This will compute all metrics possible (some depend on having loaded waveforms). 
        The metrics are:
            - isi contamination
            - firing rate
            - presence_ratio
            - amplitude_cutoff
            - principal waveform metrics
        '''
        from .metrics import isi_contamination, firing_rate, presence_ratio, amplitude_cutoff

        unitmetrics = []
        self.min_sampled_time  = np.min(self.spike_times)
        self.max_sampled_time  = np.max(self.spike_times)
        t_min = self.min_sampled_time/self.sampling_rate
        t_max = self.max_sampled_time/self.sampling_rate
        
        for iclu,ts in tqdm(zip(self.cluster_info.cluster_id.values,self),
            desc = '[{0}] Computing unit metrics'.format(self.name)):
            sp = (ts/self.sampling_rate).astype(np.float32)
            unit = dict(cluster_id = iclu,
                        isi_contamination = isi_contamination(sp, 
                                                              refractory_time=0.0015,
                                                              censored_time=0,
                                                              T = t_max-t_min), # duration of the recording
                        firing_rate = firing_rate(sp,t_min = t_min,t_max = t_max),
                        presence_ratio = presence_ratio(sp,t_min=t_min, t_max=t_max))
            if not self.spike_amplitudes is None:
                unit['amplitude_cutoff'] = amplitude_cutoff(np.take(self.spike_amplitudes,torch.nonzero(self._spike_clusters_t == iclu).flatten()))
            unitmetrics.append(unit)
        self.cluster_info = pd.merge(self.cluster_info,pd.DataFrame(unitmetrics))

        if not self.cluster_waveforms_mean is None:  # compute only if there are mean waveforms
            from .waveforms import compute_waveform_metrics
            # computes the metrics and appends to cluster_info
            clumetrics = []
            for iclu,waveforms,cluster_channel in tqdm(zip(self.cluster_info.cluster_id.values,self.cluster_waveforms_mean,self.cluster_channel),
            desc = '[{0}] Computing waveform stats'.format(self.name)):
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
        for iclu in self.cluster_info.cluster_id.values:
            yield self.get_cluster(iclu) 

    def remove_duplicate_spikes(self,overwrite_phy = False):
        from .postprocess import get_overlapping_spikes_indices
        doubled = get_overlapping_spikes_indices(self.spike_times,self.spike_clusters, self.templates_raw, self.channel_positions)
        if not len(doubled):
            return
        self.spike_times = np.delete(self.spike_times,doubled)
        self.spike_clusters = np.delete(self.spike_clusters,doubled)
        
        self._set_auxiliary()

        if not self.spike_amplitudes is None:
            self.spike_amplitudes = np.delete(self.spike_amplitudes,doubled)
        if not self.spike_positions is None:
            self.spike_positions = np.delete(self.spike_positions,doubled)
        if not self.spike_templates is None:
            self.spike_templates = np.delete(self.spike_templates,doubled)
        if not self.spike_template_amplitudes is None:
            self.spike_template_amplitudes = np.delete(self.spike_template_amplitudes,doubled)
        if not self.spike_pc_features is None:
            self.spike_pc_features = np.delete(self.spike_pc_features,doubled, axis = 0)
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
        if not self.spike_pc_features is None:
            np.save(folder/'pc_features.npy', self.spike_pc_features)
    
    def _compute_template_amplitudes_and_depths(self):
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
            templates_amplitude = self.templates_amplitude.copy()
            templates_amplitude[~np.isfinite(templates_amplitude)] = np.nanmean(templates_amplitude) # Fix for when kilosort returns NaN templates, make them the average of all templates

            # compute the center of mass (X,Y) of the templates
            from .waveforms import waveforms_position
            self.template_position,self.template_channel = waveforms_position(self.templates_raw,self.channel_positions)
            # get the spike positions and amplitudes from the average templates
            self.spike_amplitudes = np.take(templates_amplitude,self.spike_templates)*self.spike_template_amplitudes
            if not self.spike_pc_features is None or not self.template_pc_features_ind is None:
                 if self.spike_pc_features.shape[0] == len(self.spike_times):
                    self.spike_positions = None
            if self.spike_positions is None:
                # without spike features, one can estimate the position from the templates used to fit.
                self.spike_positions = self.template_position[self.spike_templates,:].squeeze()

    def get_cluster_waveforms(self,cluster_id,n_waveforms = 50):
        if not hasattr(self,'cluster_waveforms'):
            self.load_waveforms()
        if str(cluster_id) in self.cluster_waveforms.keys():
            shpe = self.cluster_waveforms[str(cluster_id)].shape[0]
            idx = np.sort(np.random.choice(range(shpe),size=min(n_waveforms,shpe),replace=False))
            return self.cluster_waveforms[str(cluster_id)][idx]

    def load_waveforms(self,parallel_pool_size = 8):
        '''Loads waveform saved in a cluster_waveforms.hdf file.'''
        if not self.folder is None:
            if (self.folder/'cluster_waveforms.hdf').exists():
                if self.cluster_waveforms is None:
                    self.cluster_waveforms = h5.File((self.folder/'cluster_waveforms.hdf'),'r')
                self.cluster_waveforms_mean = self._load_optional('cluster_mean_waveforms.npy',None) # the first is the mean
                self.cluster_waveforms_std = self._load_optional('cluster_mean_waveforms.npy',None) # the second is the std
                try:
                    if self.cluster_waveforms_mean is None: # if the average waveforms are not loaded, lets load
                        with Pool(parallel_pool_size) as pool:
                            result = [r for r in tqdm(pool.imap(partial(_mean_std_from_cluster_waveforms,
                                                                        folder = self.folder),self.cluster_info.cluster_id.values),
                                                      desc='[{0}] Computing mean waveforms'.format(self.name), total = len(self))]
                            self.cluster_waveforms_mean = np.stack([r[0] for r in result])
                            self.cluster_waveforms_std = np.stack([r[1] for r in result])
                        # save if a folder exists
                        if self.folder.exists(): # save the mean waveforms
                            np.save(self.folder/'cluster_mean_waveforms.npy',
                                    np.stack([self.cluster_waveforms_mean,
                                              self.cluster_waveforms_std]).astype(np.int16)) # the type here should be read from the data
                    else: # because we saved the std in the second element
                        self.cluster_waveforms_mean = self.cluster_waveforms_mean[0]
                        self.cluster_waveforms_std = self.cluster_waveforms_std[1]
                except:
                    del self.cluster_waveforms # close the file if it crashed.
                    self.cluster_waveforms = None
                if not self.channel_gains is None:
                    self.cluster_waveforms_mean = self.cluster_waveforms_mean*self.channel_gains
                    self.cluster_waveforms_std = self.cluster_waveforms_std*self.channel_gains
                # compute the position of each cluster and the principal channel
                from .waveforms import waveforms_position
                self.cluster_position, self.cluster_channel = waveforms_position(self.cluster_waveforms_mean, self.channel_positions)
                self.cluster_info['depth'] = self.cluster_position[:,1]
                self.cluster_info['electrode'] = self.cluster_channel

        if not hasattr(self,'cluster_waveforms'):
            raise(OSError('[{0}] - Waveforms file [cluster_waveforms.hdf] not in folder. Use the .extract_waveforms(rawdata) method.'.format(self.name)))

    def _load_required(self,file,var = None):
        '''
        Loads a required variable, if the file path does not exist returns an error, 
        unless var is specified, in that case it returns var
        '''
        if var is None:
            path = self.folder / file
            assert path.exists(), '[{0}] - {1} doesnt exist'.format(self.name,path)
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
        mwave = np.mean(waveforms_file[str(icluster)][()],axis = 0) 
        stdwave = np.std(waveforms_file[str(icluster)][()],axis = 0)#/np.sqrt(waveforms_file[str(icluster)].shape[0])
    return np.stack([mwave,stdwave])

    
    
    

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
                 sampling_rate= None,
                 channel_gains = None,
                 get_waveforms = True,
                 get_metrics = True,
                 name = 'Clusters',
                 return_samples = True,
                 load_template_features = None): #if None it will load only if needed to compute metrics, otherwise set boolean
        '''
        Object to access the spike sorting results like an array

        Inputs
        ------
            - folder (A folder with phy results)
            - (optional) spike_times
            - (optional) spike_clusters
            - (optional) channel_positions
            - get_waveforms default True

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
        # defaults in case load_template_features is False
        self.spike_template_amplitudes = None
        self.spike_templates = None
        self.templates = None
        self.whitening_matrix = None
        self.template_pc_features = None
        self.spike_pc_features = None
        self.templates_raw = None
        self.templates_amplitude = None
        self.templates_position = None
        self.spike_amplitudes = None
        self.spike_positions = None
        self.cluster_waveforms_mean = None
        self.cluster_waveforms_std = None        
        self.cluster_waveforms = None
        self.return_samples = return_samples
        self.sampling_rate = sampling_rate
        
        if load_template_features is None:
            load_template_features = True
            #check if metrics file exists
            if not self.folder is None:
                if (self.folder/'cluster_info_metrics.tsv').exists():
                    load_template_features = False # no need for templates metrics are computed
                    
        if load_template_features:  # this can take a while, disable if you don't need to estimate amplitudes or so
            self.load_template_features(spike_template_amplitudes,
                                        spike_templates,
                                        spike_pc_features,
                                        template_pc_features_ind)
            # compute the raw templates and the position of each cluster based on the template position
            self.compute_template_amplitudes_and_depths()

        self.cluster_groups = self._load_optional('cluster_group.tsv')
        

        # is there a sorting metadata file on this folder?
        metadatafile = list(Path(self.folder).glob('filtered_recording*metadata.hdf'))
        self.metadata = None
        if len(metadatafile):
            self.metadata = load_dict_from_h5(metadatafile[0])
            if self.sampling_rate is None:
                self.sampling_rate = float(self.metadata['sampling_rate'])
            if self.channel_gains is None:
                self.channel_gains = self.metadata['channel_conversion_factor'].flatten()
            self.channel_shank = self.metadata['channel_shank'].flatten()

        # load waveforms
        if get_waveforms:
            self.load_waveforms()
        # load the sampling rate?
        if self.sampling_rate is None:
            self.sampling_rate = 30000.
        # get or compute metrics
        if get_metrics:
            # computes the statistics, assumes npre as symmetric
            self.compute_statistics(npre = None, srate = self.sampling_rate)  

        self.update_cluster_info()

    def load_template_features(self,
                               spike_template_amplitudes=None,
                               spike_templates = None,
                               spike_pc_features = None,
                               template_pc_features_ind = None):
        ''' Load the template features and amplitudes '''
        # Load the amplitudes used to fit the template
        self.spike_template_amplitudes = self._load_optional('amplitudes.npy',spike_template_amplitudes)
        # load the templates used to extract the spikes
        self.templates =  self._load_optional('templates.npy')
        # load spike templates (which template was fitted) for each spike
        self.spike_templates = self._load_optional('spike_templates.npy',spike_templates)
        if self.spike_templates.max() > self.templates.shape[0]:
            self.spike_templates = self.spike_clusters
        
        if not self.spike_templates is None:
            self.spike_templates = self.spike_templates.astype(np.int64)
        # load the principle component features for each spike nspikes x featuresperchannel x nfeatures            
        self.spike_pc_features = self._load_optional('pc_features.npy',spike_pc_features)
        # template index for each PC feature
        self.template_pc_features_ind = self._load_optional('pc_feature_ind.npy',template_pc_features_ind)
        # load the whitening matrix (to correct for the templates having been whitened)
        self.whitening_matrix = self._load_optional('whitening_mat_inv.npy')
        if not self.whitening_matrix is None:
            self.whitening_matrix = self.whitening_matrix.T
            
    def extract_waveforms(self,data, chmap = None,
                          max_n_spikes = 1000,
                          npre = 45, npost = 45,
                          chunksize = 20,
                          filter_par = dict(order = 3,
                                            flow = 300/15000,
                                            fhigh = 10000/15000,
                                            car = True),
                          save_folder_path = None):
        '''
        Extract waveforms from raw data
        
        Parameters
        ----------
        data
        chmap
        max_n_spikes
        npre
        npost
        chunksize
        save_folder_path
        
        Returns
        -------
        waveforms: dict 
            keys are cluster ids
        '''
        if chmap is None:
            if hasattr(data,'channel_info'):
                chmap = data.channel_info.channel_idx.values
            else:
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
        if not filter_par is None:
            p_bar = tqdm(desc = 'Filtering waveforms',total = len(self.cluster_id))
        for iclu,w in zip(self.cluster_id,mwaves):
            if filter_par is None:
                waveforms[int(iclu)] = w
            else:
                waveforms[int(iclu)] = filter_waveforms(w,**filter_par)
                p_bar.update(1)
        if not filter_par is None:
            del p_bar
        if not save_folder_path is None:
            print(save_folder_path)
            if Path(save_folder_path).exists():
                print('Saving waveforms to {}'.format(save_folder_path))
                save_dict_to_h5(save_folder_path/'cluster_waveforms.hdf',waveforms)
            else:
                print(f'Folder not found {savefolder}')
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
        '''
        Run this when you change the cluster_info DataFrame to update the object 
        '''
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
        if self.return_samples:
            if len(sp) == 1:
                return sp[0]
            else:
                return sp
        else:
            if len(sp) == 1:
                return sp[0].astype(np.float32)/self.sampling_rate
            else:
                return [s.astype(np.float32)/self.sampling_rate for s in sp]

    def compute_statistics(self,srate = 30000.,npre = None,recompute = False):
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

        cluster_info_metrics = self._load_optional('cluster_info_metrics.tsv',None)
        if not cluster_info_metrics is None and not recompute:
            self.cluster_info = cluster_info_metrics
            return
        from .metrics import isi_contamination, firing_rate, presence_ratio, amplitude_cutoff
        
        unitmetrics = []
        self.min_sampled_time  = np.min(self.spike_times)
        self.max_sampled_time  = np.max(self.spike_times)
        t_min = self.min_sampled_time/self.sampling_rate
        t_max = self.max_sampled_time/self.sampling_rate
        if self.spike_amplitudes is None:
            # try to get them..
            self.compute_template_amplitudes_and_depths()
        # clean-up cluster info
        
        self.cluster_info = pd.DataFrame(
            dict(cluster_id = self.cluster_info['cluster_id'].values))
        self.load_waveforms(reload=recompute) # loads electrode depth and so to cluster_info
        for iclu,ts in tqdm(zip(self.cluster_info.cluster_id.values,self),
            desc = '[{0}] Computing unit metrics'.format(self.name)):
            sp = ts
            if self.return_samples:
                sp = sp.astype(np.float32)/self.sampling_rate
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
            # compute the position of each cluster and the principal channel
            from .waveforms import waveforms_position
            self.cluster_position, self.cluster_channel = waveforms_position(self.cluster_waveforms_mean,
                                                                             self.channel_positions)
            self.cluster_info['depth'] = self.cluster_position[:,1]
            self.cluster_info['electrode'] = self.cluster_channel
            if hasattr(self,'channel_shank'):
                self.cluster_info['shank'] = [
                    self.metadata['channel_shank'][
                        self.channel_map.flatten() == c].flatten()[0]
                    for c in self.cluster_channel.flatten()]

            from .waveforms import compute_waveform_metrics
            # computes the metrics and appends to cluster_info
            clumetrics = []
            for iclu,waveforms,cluster_channel in tqdm(zip(self.cluster_info.cluster_id.values,
                                                           self.cluster_waveforms_mean,
                                                           self.cluster_channel),
            desc = '[{0}] Computing waveform stats'.format(self.name)):
                wave = waveforms[:,cluster_channel]
                if npre is None:
                    npre = int(wave.shape[0]/2)
                clumetrics.append(dict(cluster_id = iclu,
                                **compute_waveform_metrics(wave,npre=npre,srate=srate)))
            clumetrics = pd.DataFrame(clumetrics)
            activeidx = estimate_active_channels(self.cluster_waveforms_mean)
            nactive_channels = np.array([len(a) for a in activeidx])
            clumetrics['n_active_channels'] = nactive_channels
            clumetrics['active_channels'] = activeidx
            self.cluster_info = pd.merge(self.cluster_info,clumetrics)
        if not self.cluster_info is None and not self.cluster_waveforms_mean is None: # save if the folder exists to make it faster to load
            if self.folder.exists():
                self.cluster_info.to_csv(self.folder/'cluster_info_metrics.tsv',sep = '\t',index = False)
        return self.cluster_info
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
    
    def compute_template_amplitudes_and_depths(self):
        '''

        Compute the amplitude and depths(positions) of each spike from the template fitting

        This takes a while.

        '''

        if self.templates is None:
            # then try to load the templates
            self.load_template_features()

        if (not self.templates is None and 
            not self.whitening_matrix is None and 
            not self.channel_positions is None):
            # TODO: move to a separate function
            #
            # the raw templates are the dot product of the templates by the whitening matrix
            self.templates_raw = np.dot(self.templates,self.whitening_matrix)
            # compute the peak to peak of each template
            templates_peak_to_peak = (self.templates_raw.max(axis = 1) - self.templates_raw.min(axis = 1))
            # the amplitude of each template is the max of the peak difference for all channels
            self.templates_amplitude = templates_peak_to_peak.max(axis=1)
            templates_amplitude = self.templates_amplitude.copy()
            # Fix for when kilosort returns NaN templates, make them the average of all templates
            templates_amplitude[~np.isfinite(templates_amplitude)] = np.nanmean(templates_amplitude)
            # compute the center of mass (X,Y) of the templates
            from .waveforms import waveforms_position
            self.template_position,self.template_channel = waveforms_position(self.templates_raw,self.channel_positions)
            # get the spike positions and amplitudes from the average templates
            self.spike_amplitudes = np.take(templates_amplitude,self.spike_templates)*self.spike_template_amplitudes
            # if there is a spike_positions.npy file, take the positions from there.
            self.spike_positions = self._load_optional('spike_locations.npy',None)
            if self.spike_positions is None: # compute from pc_features
                if not self.spike_pc_features is None or not self.template_pc_features_ind is None:
                    if self.spike_pc_features.shape[0] == len(self.spike_times):
                        self.spike_positions = None
                        # Compute the spike depth from the features
                        self.spike_positions = estimate_spike_positions_from_features(
                            self.spike_templates,
                            self.spike_pc_features,
                            self.template_pc_features_ind,
                            self.channel_positions)
                    else:
                        print('''[0] warning spike_pc_features does not have the same size as spike_amplitudes.
                        
Spike depths will be based on the template position. Re-sort the dataset to fix {1}'''
                              .format(self.name,self.folder))
                
            if self.spike_positions is None:
                # without spike features, one can estimate the position from the templates used to fit but it is not great
                print('[Clusters] warning: taking spike positions from average template position.')
                self.spike_positions = self.template_position[self.spike_templates,:].squeeze()
                
    def get_cluster_waveforms(self,cluster_id,n_waveforms = 50):
        if not hasattr(self,'cluster_waveforms'):
            self.load_waveforms()
        if str(cluster_id) in self.cluster_waveforms.keys():
            shpe = self.cluster_waveforms[str(cluster_id)].shape[0]
            idx = np.sort(np.random.choice(range(shpe),size=min(n_waveforms,shpe),replace=False))
            return self.cluster_waveforms[str(cluster_id)][idx]

    def load_waveforms(self,parallel_pool_size = 8,reload = False):
        '''Loads waveform saved in a cluster_waveforms.hdf file.'''
        if not self.folder is None:
            self.cluster_waveforms_mean = self._load_optional('cluster_mean_waveforms.npy',None) # the first is the mean
            self.cluster_waveforms_std = self._load_optional('cluster_mean_waveforms.npy',None) # the second is the std
            if not self.cluster_waveforms_mean is None:
                # because we saved the std in the second element
                self.cluster_waveforms_mean = self.cluster_waveforms_mean[0]
                self.cluster_waveforms_std = self.cluster_waveforms_std[1]
                if self.cluster_waveforms_mean.shape[0] != len(self.cluster_info):
                    # then we need to re-compute the mean waveforms 
                    reload = True
            if (self.folder/'cluster_waveforms.hdf').exists():
                if self.cluster_waveforms is None:
                    self.cluster_waveforms = h5.File((self.folder/'cluster_waveforms.hdf'),'r')
                try:
                    if self.cluster_waveforms_mean is None or reload: # if the average waveforms are not loaded, lets load
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
                except Exception as err:
                    print(err)
                    del self.cluster_waveforms # close the file if it crashed.
        
        if not hasattr(self,'cluster_waveforms') or self.cluster_waveforms is None:
            print('[{0}] - Waveforms file [cluster_waveforms.hdf] not in folder. Use the .extract_waveforms(rawdata) method.'.format(self.name))

        if not self.channel_gains is None and not self.cluster_waveforms_mean is None:
            # scale the waveforms
            self.cluster_waveforms_mean = self.cluster_waveforms_mean*self.channel_gains
            self.cluster_waveforms_std = self.cluster_waveforms_std*self.channel_gains

    def plot_drift_map(self,**kwargs):
        if self.spike_positions is None:
            self.compute_template_amplitudes_and_depths() 
        from .viz import plot_drift_raster
        plot_drift_raster(self.spike_times.astype(np.float32)/self.sampling_rate,
                          self.spike_positions[:,1],
                          self.spike_amplitudes,**kwargs)
        
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

    
    
def estimate_spike_positions_from_features(spike_templates,spike_pc_features,template_pc_features_ind,channel_positions,consider_feature=0):
    '''
    Estimates the spike 2d location based on a feature e.g the PCs.
    
    This is adapted from the cortexlab/spikes repository to estimate spikes based on the PC features.
    TODO: make this work for voltage as well.

    Parameters
    ----------
    spike_templates: nspikes tesmplates used for each spike
    spike_pc_features: nspikes x nfeatures x nchannels
    template_pc_features_ind: indice of the channels for the templates nchannels
    channel_positions: position of each channel
    consider_feature: feature to consider

    Returns
    -------
    spike_locations: nspikes

    Joao Couto - spks 2023
    '''
    # channel index for each feature
    feature_channel_idx = torch.index_select(numpy_to_tensor(template_pc_features_ind),dim=0,
                                             index = numpy_to_tensor(spike_templates).flatten())
    # 2d coordinates for each channel feature
    feature_coords = torch.index_select(numpy_to_tensor(channel_positions),dim=0,
                                        index=feature_channel_idx.flatten()).reshape([*feature_channel_idx.shape,*channel_positions.shape[1:]])
    # ycoords of those channels?
    pc_features = numpy_to_tensor(spike_pc_features[:,consider_feature].squeeze())**2 # take the first pc for the features
    spike_locations = (torch.sum(feature_coords.permute((2,0,1))*pc_features,dim=2)/torch.sum(pc_features,dim=1)).t()
    return tensor_to_numpy(spike_locations)


def filter_waveforms(waveforms,flow = 300/15000, fhigh = 10000/15000,order = 3,car = True):
    ''' 
    Filter spike waveforms.
    Parameters
    ----------
    waveforms Nwaves x time samples x nchannels
    flow
    fhigh
    order
    car

    Returns
    -------
    filtered waveforms
    
    Joao Couto - spks 2023
    '''
    if not flow is None or fhigh is None:
        b,a = signal.butter(order,(flow,fhigh),'pass')
        waveforms = signal.filtfilt(b,a,waveforms,axis = 1)
    if car:
        waveforms = (waveforms.T - np.median(waveforms,axis=2).T).T
    return waveforms

def filter_waveforms_gpu(waveforms,low = 300/15000, high = 10000/15000,order = 3,car = True,device = 'cpu',
                         padlen = 80):
    '''
    this is work in progress, not tested.    Joao Couto - Nov 2023
    '''
    T = numpy_to_tensor(waveforms.astype(np.float32),device = device)
    b,a = butter(order,[low, high], btype='bandpass')
    aa = torch.from_numpy(np.array(a,dtype='float32')).to(device)
    bb = torch.from_numpy(np.array(b,dtype='float32')).to(device)
    X = torch.zeros_like(T)
    for i in range(T.shape[0]):
        t = torch.nn.functional.pad(T[i].T,(padlen,padlen),'replicate') # apply padding for filter
        X[i] = torchaudio.functional.filtfilt(t,aa,
                                                  bb,clamp=False)[:,padlen:-padlen].T
    if car:
        X = (X.permute(2,0,1) - torch.median(X,axis=2).values).permute(1,2,0)
        
    return tensor_to_numpy(X).astype(waveforms.dtype)




class MultiprobeClusters():
    def __init__(self,sessions, get_waveforms = True, return_samples = False):
        '''
        Checks if data are sorted for a set of sessions and loads data for all probes.
        It will also compute the alignments of spiketimes of all probes to the first probe and generate the sync interpolation functions
        '''
        self.clusters = []
        self.return_samples = return_samples
        infos = []
        from .io import list_sorting_result_paths
        for s in sessions:
            sortings = list_sorting_result_paths(s)
            if len(sortings):
                for p,sfolder in enumerate(sortings):
                    self.clusters.append(Clusters(sfolder[0],name = f'probe{p}',
                                                  get_waveforms = get_waveforms))
                    self.clusters[-1].cluster_info['probe']  = p
                    infos.append(self.clusters[-1].cluster_info)
                break # loads only from the first session
        if not len(infos):
                print(f'[MultiprobeCluster] - Could not load {sessions}')
                return None
        self.cluster_info = pd.concat(infos)
        self.update_cluster_info()
        self.sync_multiprobe_recordings()
        
    def update_cluster_info(self): 
        self.__dict__.update(dict(zip(list(self.cluster_info), self.cluster_info.to_numpy().T)))

    def sync_multiprobe_recordings(self,probe_sync_bit = 6): # does the sync_bit change between systems?
        if hasattr(self,'master_syncs'):
            return # data were aligned already
        # probes get aligned to probe0
        # probes that share a headstage are aligned by definition (share a clock)
        # This uses the sync data collected using the metadata, make it work when these files dont exist.
        # lets align each 'file' separately; the offsets are the same for all probes
        file_offsets = self.clusters[0].metadata['file_offsets']
        self.master_syncs = [] # there is a master sync per recorded file.
        for ifile,offsets in enumerate(file_offsets):
            probe_sync = [c.metadata[f'file{ifile}_sync_onsets'] for c in self.clusters]
            probe_sync = [p[probe_sync_bit] for p in probe_sync]
            probe_sync = [p[(p>offsets[0]) & (p<offsets[1])] for p in probe_sync]
            master_sync = probe_sync[0]
            sync_func = [interp1d(p, master_sync, fill_value = 'extrapolate') for p in probe_sync]
            # apply the corrections to the spike times of every cluster
            for icluster,(c,f) in enumerate(zip(self.clusters,sync_func)):
                cidx = (c.spike_times>offsets[0]) & (c.spike_times<offsets[1])
                self.clusters[icluster].spike_times[cidx] = f(c.spike_times[cidx]).astype(c.spike_times.dtype)
                self.clusters[icluster].sampling_rate = self.clusters[0].sampling_rate # adjust the sampling rate of each cluster
            self.master_syncs.append(master_sync)
            self.sampling_rate = self.clusters[0].sampling_rate
        # plt.figure()
        # plt.plot(probe_sync[0],(probe_sync[2]-probe_sync[0]),'ko')
        # plt.plot(probe_sync[0],sync_func[2](probe_sync[2])-probe_sync[0],'ro')
        
    def __getitem__(self, index):
        ''' returns the spiketimes for a set of clusters'''
        if type(index) in [int,np.int64,np.int32]:
            index = [index]
        if type(index) in [slice]:
            index = np.arange(*index.indices(len(self)))
        sp = []
        for idx,item in self.cluster_info[index].iterrows():
            sp.append(self._get_cluster(item))  
        if self.return_samples:
            if len(sp) == 1:
                return sp[0]
            else:
                return sp
        else:
            if len(sp) == 1:
                return sp[0].astype(np.float32)/self.sampling_rate
            else:
                return [s.astype(np.float32)/self.sampling_rate for s in sp]
            
    def _get_cluster(self,item):
        '''gets the cluster from each 'clusters' object'''
        probe = item.probe
        cluster_id = item.cluster_id
        return self.clusters[probe].get_cluster(cluster_id)
    def __len__(self):
        return len(self.cluster_info)

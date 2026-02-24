from .utils import *
import torchaudio
from scipy.signal import butter
from .spikeglx_utils import load_spikeglx_binary, load_spikeglx_mtsdecomp

SPIKEGLX_FILE_EXTENSION = '.bin'
MTSCOMP_FILE_EXTENSION = '.cbin'


@torch.no_grad()
def filtfilt_chunk(chunk,a,b,global_car=False, return_gpu = True, device=None, padlen = 150):
    '''
    Filter a chunk of data in the time domain using filter coeffients.
      - chunk are [TIMExCHANNELS] 
      - a and b are the coefficients
    '''
    device = check_cuda(device)
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
    X = torchaudio.functional.filtfilt(T.to(torch.float32),aa,
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

def global_car_gpu(chunk, device = None,return_gpu = False):
    device = check_cuda(device)
    if isinstance(chunk,np.ndarray):
        dtype = chunk.dtype
        chunk = torch.from_numpy(chunk,device = device)
    if chunk.shape[1]<6: # only compute the median if the shape is larger than 6 channels
        X = chunk.T
    else:
        X = chunk.T-torch.median(chunk,axis=1).values
    if return_gpu:
        return X.T
    else:
        return tensor_to_numpy(X.T)

def bandpass_filter_gpu(data,sampling_rate, lowpass, highpass, order = 3, device = None, return_gpu = True):
    '''
    bandpass filter using pytorch
    '''
    sratecoeff = sampling_rate/2.
    b,a = butter(order,[lowpass/sratecoeff, highpass/sratecoeff], btype='bandpass')

    return filtfilt_chunk(data, a , b, device = device, return_gpu = return_gpu)

def phase_shift_gpu(X,sample_shifts,device = None, padlen = 50, return_gpu = True):
    ''' 
    Compensate for the phase shift induced during digitization.

    Adapted for pytorch from spike interface. 
    '''
    device = check_cuda(device)
    if isinstance(X,np.ndarray):
        dtype = X.dtype
        X = torch.from_numpy(X,device = device)
    if sample_shifts is None:
        # skipping.
        return X
    # apply padding
    T = torch.nn.functional.pad(X.to(torch.float32).T,(padlen,padlen),'reflect').T # apply padding for filter
    xf = torch.fft.rfft(T,axis = 0)
    n = T.shape[0]
    if n % 2 == 0:
        # n is even sig_f[-1] is nyquist and so pi
        omega = torch.linspace(0, np.pi, xf.shape[0],device=device)
    else:
        # n is odd sig_f[-1] is exactly nyquist!! we need (n-1) / n factor!!
        omega = torch.linspace(0, np.pi * (n - 1) / n, xf.shape[0],device = device)
    # broadcast omega and sample_shifts depend the axis

    if X.shape[1] == len(sample_shifts)-1:
        # then you passed a sync channel also, make it not fail
        sample_shifts = np.hstack([sample_shifts,0])
    shifts = omega[:, np.newaxis] * numpy_to_tensor(sample_shifts[np.newaxis, :], device = device)
    xf = torch.fft.irfft(xf * torch.exp(-1j * shifts), n=n, axis=0)
    xf = xf[padlen:-padlen,:]
    if return_gpu:
        return xf
    else:
        return tensor_to_numpy(xf)


default_filter_pipeline_par = [dict(function = 'bandpass_filter_gpu',
                                    sampling_rate = 30000,
                                    lowpass = 300,
                                    highpass = 12000,
                                    return_gpu = True),
                               dict(function = 'phase_shift_gpu',
                                    sample_shifts = None,
                                    return_gpu = True),
                               dict(function = 'global_car_gpu',
                                    return_gpu = False)]

def parse_filter_pipeline(filterlist):
    '''
    Filter parameters are a list of dictionaries. 
    Parameters:
        filterlist: list
            Each dictionary has a "function" key that specifies which function to use. 
    
    Returns: 
        filters: list
            List of functions that take the data as input and can be applied sequentially.
            
    '''
    from spks.raw import (bandpass_filter_gpu,
                          bandstop_filter_gpu,
                          phase_shift_gpu,
                          global_car_gpu) # import filters from here
    functions = []
    for f in filterlist:
        func = f['function']
        par = f.copy()
        del par['function']
        functions.append(partial(eval(func),**par))
    return functions

def shifts_from_adc_channel_groups(adc_channel_groups):
    '''
    Compute the shifts in ADC groups
    
Example:
channel_ids,tshifts = shifts_from_adc_channel_groups(dat.metadata[0]['adc_channel_groups']  )
    '''
    shifts = []
    nsamples = len(adc_channel_groups) 
    for i,group in enumerate(adc_channel_groups):
        shifts.append(np.ones(len(group))*i/nsamples)
    channel_ids,tshifts = np.hstack(adc_channel_groups),np.hstack(shifts)
    isort = np.argsort(channel_ids)
    return channel_ids[isort],tshifts[isort]

def bandstop_filter_gpu(data,sampling_rate, lowpass, highpass, order = 3, device = None, return_gpu = True):
    '''
    bandpass filter
    '''
    sratecoeff = sampling_rate/2.
    b,a = butter(order,[lowpass/sratecoeff, highpass/sratecoeff], btype='bandstop')
    return filtfilt_chunk(data, a , b, device = device, return_gpu = return_gpu)

class RawRecording(object): 
    def __init__(self,files, 
                 filter_pipeline_par = [dict(**d) for d in
                                        default_filter_pipeline_par],
                 return_preprocessed = True,
                 device = None,  #TODO: make that the functions can make use of this. Right now it always uses the cuda if available..
                 return_voltage = False, **kwargs):
        '''
        Pretend that the recordings are concatenated.
        There is a limit to the chunk size because of how processing is done (gpu).
        '''
        # load the files, can be compressed bin or bin
        # get the recording duration by iterating through the files
        self.device = device
        self.files = files
        self.current_index = None
        self.nsamples = None
        self.offsets = []
        self.metadata = []
        self.conversion_factor = 1.
        self.dtype = np.int16
        self.return_preprocessed = return_preprocessed
        self.return_voltage = return_voltage
        self._init_parameters()
        self._parse_filter_pipeline_par(filter_pipeline_par)

    def _parse_filter_pipeline_par(self,filter_pipeline_par, channels = None):
        ''' 
Gets the sampling rate into all filters that need it and initializes filter functions
        '''
        self.filter_pipeline = []
        self.filter_pipeline_par = filter_pipeline_par

        for i,f in enumerate(self.filter_pipeline_par):
            for k in f.keys():
                if k == 'sampling_rate':
                    self.filter_pipeline_par[i][k] = self.sampling_rate
                if (k == 'sample_shifts' and
                    'adc_channel_groups' in self.metadata[0].keys()):
                    # TODO handle when there are more channels than possible
                    self.filter_pipeline_par[i][k] = shifts_from_adc_channel_groups(self.metadata[0]['adc_channel_groups'])[1]
                    if not channels is None:
                        tt = self.filter_pipeline_par[i][k]
                        if np.max(channels)>len(tt)-1:
                            tt = np.zeros(np.max(channels)+1)
                            tt[:len(self.filter_pipeline_par[i][k])] = self.filter_pipeline_par[i][k]
                        self.filter_pipeline_par[i][k] = tt[channels]
        self.filter_pipeline = parse_filter_pipeline(self.filter_pipeline_par)
        
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
            elif type(index[1]) in [int,np.int32, np.int64]:
                idx2 = [index[1]]
            elif type(index[1]) in [list,np.array,np.ndarray]:
                idx2 = [i for i in index[1]]
            index = index[0]
        if type(index) is slice:
            idx1 = range(*index.indices(self.shape[0]))#start, index.stop, index.step)
        elif type(index) in [int,np.int32, np.int64]: # just a timesample
            idx1 = [index]
        else: # np.array?
            idx1 = index
        if idx2 is None:
            idx2 = range(self.shape[1])
        self._parse_filter_pipeline_par(self.filter_pipeline_par,channels = idx2)
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
        for ifile,(o,f) in enumerate(self.file_sample_offsets):
            buffidx = np.where((selidx>=o) & (selidx<f))[0]
            if not len(buffidx):
                continue
            self._set_current_buffer(ifile)
            fileidx = selidx[buffidx]-o
            #print(f'requested {selidx[0]}:{selidx[-1]} which is file {ifile} {fileidx[0]}:{fileidx[-1]}')
            tmp = self.buffers[ifile][fileidx][:,rows]
            if len(tmp):
                if return_preprocessed:
                    for func in self.filter_pipeline:
                        tmp = func(tmp)
                buffer[buffidx,:] = tmp
        if return_voltage:
            gains = np.ones(self.shape[1],dtype=np.float32)
            gains[self.channel_info.channel_idx.values.astype(int)] = self.channel_info.conversion_factor.values
            return (buffer.astype(np.float32) * gains[rows])
        return buffer

    def _load_buffers(self):
        self.buffers = []
        for ifile,file in enumerate(self.files):
            if self.file_extensions[ifile] == SPIKEGLX_FILE_EXTENSION: #TODO: pass a loading function when initializing RawRecording?
                self.buffers.append(load_spikeglx_binary(file)[0])
            elif self.file_extensions[ifile] == MTSCOMP_FILE_EXTENSION:
                self.buffers.append(load_spikeglx_mtsdecomp(file)[0])

    def _set_current_buffer(self,ibuffer):
        if not self.current_index == ibuffer:
            self.current_index = ibuffer
            self.current_pointer = self.buffers[self.current_index]

    def _init_parameters(self):
        ''' This function depends on the reader. It should populate the parameters of the object.'''
        self.file_extensions = []
        for ifile,f in enumerate(self.files):
            if not os.path.exists(f):
                raise(OSError('[RawRecording] - {0} file not found.'.format(f)))
            self.file_extensions.append(Path(f).suffix)
            if self.file_extensions[ifile] == SPIKEGLX_FILE_EXTENSION:
                self.current_pointer,meta = load_spikeglx_binary(f)
            elif self.file_extensions[ifile] == MTSCOMP_FILE_EXTENSION:
                self.current_pointer,meta = load_spikeglx_mtsdecomp(f)
            self.offsets.append(self.current_pointer.shape[0])
            self.metadata.append(meta)
            if ifile == 0:
                self.dtype = self.current_pointer.dtype
                self.sampling_rate = meta['sRateHz']
                self.channel_info = pd.DataFrame(
                    zip(meta['channel_idx'],meta['coords'],meta['channel_shank'],meta['conversion_factor_microV']),
                    columns = ['channel_idx','channel_coord','channel_shank','conversion_factor'])
        self._load_buffers()
        self._set_current_buffer(0)
        self._set_shape_and_offsets()

    def _set_shape_and_offsets(self, offsets = None):
        # this can be used to trim the files but it is undocumented for the moment
        if not offsets is None:
            self.offsets = offsets
        self.shape = (sum(self.offsets),self.current_pointer.shape[1])
        if len(self.offsets)>1:
            self.file_sample_offsets = np.vstack([np.hstack([[0],np.cumsum(self.offsets)[:-1]]),
            np.hstack([np.cumsum(self.offsets)])]).T
        else:
            self.file_sample_offsets = [[0,self.offsets[0]]]

    def extract_syncs(self, sync_channel = -1, unpack = True, chunksize = 600000):
        '''Syncs are extracted from the sync channel and converted into onsets and offsets.'''
        from tqdm import tqdm
        sync_onsets = []
        sync_offsets = []
        for i,f in enumerate(self.files):
            trace = binary_read_single_channel(f,channel_idx=-1)
            from spks.sync import unpackbits_gpu
            onsets,offsets = unpackbits_gpu(trace)
            sync_onsets.append(onsets)
            sync_offsets.append(offsets)
        self.sync_onsets = sync_onsets
        self.sync_offsets = sync_offsets
        return sync_onsets,sync_offsets

    def get_n_jobs(self, chunksize = 30000*2,nchannels = None, required_per_worker = 4*26):
        n_jobs = 8
        if torch.cuda.is_available():
            # This  depends on which preprocessing is done.. For the fft we need more memory
            if nchannels is None:
                nchannels = self.shape[1]
            n_jobs = int(np.floor(torch.cuda.mem_get_info()[0]/(chunksize*required_per_worker*nchannels)))
        return n_jobs

    def to_binary(self, filename, channels = None, processed = True, 
                  chunksize = 30000*2, sync_channel = -1, 
                  get_channels_mad = True,
                  n_jobs = None,
                  filter_pipeline_par = [dict(**d) for d in
                                         default_filter_pipeline_par],
                  chunks = None):
        # create a binary file
        '''
        Exports to binary file and a channelmap.
        '''
        if not str(filename).endswith('.bin'): 
            filename += '.bin'
        from .sync import unpackbits_gpu
        if chunks is None:
            # allows passing specific chunks, for instance to trim down the file
            chunks = chunk_indices(self,chunksize = chunksize)
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        if channels is None:
            channels = np.arange(self.shape[1], dtype = int)
        self._parse_filter_pipeline_par(filter_pipeline_par, channels = channels)
        filter_pipeline_par = self.filter_pipeline_par
        out = np.memmap(filename,
                        dtype = self.dtype,
                        mode = 'w+',
                        shape=(self.shape[0],len(channels)))
        from joblib import Parallel,delayed
        from tqdm import tqdm        
        # get the number of jobs depending on the available cuda size
        if n_jobs is None:
            n_jobs = self.get_n_jobs(chunksize,nchannels = len(channels))
        # print(f'n_jobs = {n_jobs}')
        with Parallel(n_jobs = n_jobs) as pool:
            # Run a parallel pool that writes the binary
            sync = pool(delayed(_write_chunk_from_files)(
                self.files, chunk, out,
                channels = channels,
                sync_channel = sync_channel,
                filter_pipeline_par = filter_pipeline_par,
                offsets = self.offsets)
                        for chunk in tqdm(chunks,
                                          desc = 'Exporting binary'))
            # free all gpu jobs
            pool(delayed(lambda x:free_gpu())(i) for i in range(n_jobs))
        # close all pools in case they are still running
        free_gpu()
        out.flush()
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)

        # save metadata
        nchannels = len(channels)
        channel_positions = []
        conversion_f = []
        channel_shank = []
        channels = self.channel_info.channel_idx.values.flatten()
        for c in [c for c in channels]:
            gain = self.channel_info.conversion_factor[self.channel_info.channel_idx == c].values
            coord = self.channel_info.channel_coord[self.channel_info.channel_idx == c].values
            shank = self.channel_info.channel_shank[self.channel_info.channel_idx == c].values
            if len(coord):
                channel_positions.append([c for c in coord[0]])
                conversion_f.append(gain)
                channel_shank.append(shank)
            else:
                channel_positions.append([None,None])
                conversion_f.append(1.0)
                channel_shank.append(0)
        metadata = dict(sampling_rate = self.sampling_rate,
                        original_channels = [c for c in channels],
                        nchannels = nchannels,
                        channel_idx = [c for c in np.arange(nchannels,dtype=int)],
                        channel_coords = np.stack(channel_positions).squeeze(),
                        channel_conversion_factor = np.stack(conversion_f).flatten(),
                        channel_shank = np.stack(channel_shank).flatten(),
                        file_offsets = self.file_sample_offsets,
                        filenames = [os.path.basename(f) for f in self.files])
        # sync data
        if not sync_channel is None:
            sync = np.hstack(sync)
            for ifile,(o,f) in tqdm(enumerate(self.file_sample_offsets),
                                    desc = 'Unpacking sync channel'):
                onsets,offsets = unpackbits_gpu(sync[o:f-1],device = self.device)
                metadata[f'file{ifile}_sync_onsets'] = onsets
                metadata[f'file{ifile}_sync_offsets'] = offsets
            free_gpu()            
        if get_channels_mad: # median absolute deviation of the first 30 seconds
            mad_int16 = [m for m in mad(out[:30000*30,:])]
            metadata['channel_mad_int16'] = mad_int16
        # del out
        save_dict_to_h5(str(filename).replace('.bin','.metadata.hdf'), metadata)
        return out, metadata



def _write_chunk_from_files(files, chunk, outputmmap,
                            channels = None, 
                            filter_pipeline_par = None,
                            sync_channel = -1,
                            offsets = None):
    '''
    Support function for writing chunks a memory mapped file.
    Example usage with joblib:
    

    '''
    dat = RawRecording(files,return_preprocessed=False)
    if not offsets is None: # in case the shape is trimmed
        dat._set_shape_and_offsets(offsets)
    filter_pipeline = parse_filter_pipeline(filter_pipeline_par)
    buf = dat[chunk[0]:chunk[1]]
    if not sync_channel is None:
        sync_channel = buf[:,sync_channel]
    if not channels is None: # select channels
        buf = buf[:,channels]
    # process only for the selected channels
    for func in filter_pipeline:
        buf = func(buf)
    outputmmap[chunk[0]:chunk[1],:] = buf[:]
    #print(f'saving {chunk[0]}:{chunk[1]}')
    del dat
    return sync_channel        

def load_spks_binary_to_spikeinterface(binary_fname,metadata = None):
    ''' Loads a RawRecording binary to spikeinterface as a Recording object.
    metadata is a dict with fields "sampling_rate", "nchannels", "channel_coords" and "channel_shank"
    '''
    import spikeinterface.full as si
    if metadata is None:
        metadata_fname = str(binary_fname).replace('.bin','.metadata.hdf')
        metadata = load_dict_from_h5(metadata_fname)

    rec = si.read_binary(binary_fname,sampling_frequency=metadata['sampling_rate'],
                     dtype = np.int16, num_channels = metadata['nchannels'],
                     is_filtered = True)
    rec.is_filtered = True

    rec.set_channel_locations(metadata['channel_coords'].astype(int))
    rec.set_channel_groups(metadata['channel_shank'])
    return rec

def dredge_motion_correct_across_sessions(binaryfilepath,metadata,channel_info,segment_duration = 3, n_jobs = 10):
    seg_duration = int(segment_duration*60*metadata['sampling_rate'])

    shanks = np.unique(channel_info['channel_shank'].values)
    job_kwargs = dict(chunk_duration="1s", n_jobs=n_jobs, progress_bar=True)
    tmp_folder = Path(binaryfilepath).parent
    concat_offsets = np.cumsum([0]+[seg_duration for o in metadata['file_offsets']])

    from spikeinterface.preprocessing import bandpass_filter
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    from spikeinterface.sortingcomponents.peak_selection import select_peaks
    from spikeinterface.sortingcomponents.peak_localization import localize_peaks
    from spikeinterface.sortingcomponents.motion import estimate_motion, interpolate_motion, correct_motion_on_peaks

    for shank in shanks: # do the processing per shank 
        chan_info = channel_info[channel_info['channel_shank'].values == shank]

        #concatenated 5 min
        trimmed_filename = Path(tmp_folder)/f'temporary_rec_{shank}.trimmed.bin'
        trimmed_filename.parent.mkdir(exist_ok = True,parents = True)
        trimmed_meta = dict(sampling_rate =  metadata["sampling_rate"],
                            nchannels = len(chan_info),
                            channel_coords = np.stack(chan_info['channel_coord'].values),
                            channel_shank = np.stack(chan_info['channel_shank']))
        trimmed_bin = np.memmap(trimmed_filename,
                                dtype = raw_rec.dtype,
                                mode = 'w+',
                                shape=(concat_offsets[-1],len(chan_info)))
        # read the whole file with spike interface, then split in segments per session and run dredge on that.
        entire_rec = load_spks_binary_to_spikeinterface(binaryfilepath)
        motion_objs = []
        peaks_sample = []
        peak_locations_sample = []
        for irec,(o,e) in enumerate(metadata['file_offsets']):
            rec = entire_rec.frame_slice(start_frame=o,end_frame=e).select_channels(chan_info[channel_idx.values])
            rec = bandpass_filter(recording=rec, freq_min=300., freq_max=5000.)
            recording_corrected, motion_info = si.correct_motion(rec, preset='dredge',
                                                                output_motion_info=True, **job_kwargs)
            motion_objs.append(motion_info)

            peak_locations = correct_motion_on_peaks(motion_info['peaks'], motion_info['peak_locations'], motion_info['motion'],rec)
            # prepare the correction 
            concat_offset = concat_offsets[irec]
            for no,ne in tqdm(chunk_indices(np.arange(concat_offsets[1])),desc = 'Saving trimmed traces'):
                trimmed_bin[concat_offset+no:concat_offset+ne] = recording_corrected.get_traces(segment_index=0,
                                                                                                start_frame = no, 
                                                                                                end_frame=ne).round().astype('int16')
        trimmed_rec = load_spks_binary_to_spikeinterface(trimmed_filename,trimmed_meta)
        _ , motion_info_trimmed = si.correct_motion(trimmed_rec, preset='dredge',
                                                    output_motion_info=True, **job_kwargs) 
        # correct the motion for each
        t = motion_info_trimmed['motion'].temporal_bins_s[0]
        d = motion_info_trimmed['motion'].displacement[0]
        for irec in range(len(concat_offsets)-1):
            t = motion_info_trimmed['motion'].temporal_bins_s[0]
            d = motion_info_trimmed['motion'].displacement[0]
            ii = (t> concat_offsets[irec]/metadata['sampling_rate']) & (t< concat_offsets[irec+1]/metadata['sampling_rate']) 
            d = np.median(d[ii],axis = 0)
            motion_objs[irec]['motion'].displacement[0] = (motion_objs[irec]['motion'].displacement[0] + d)
        # correct the offset in the raw concatenated file
        #                                                
        def _interpolate_rec_parallel(ori_start,new_start,new_end,out,recording_corrected,channels):
            out[ori_start+new_start:ori_start+new_end,channels] = recording_corrected.get_traces(segment_index=0,
                                                                            start_frame = new_start, 
                                                                            end_frame=new_end).round().astype('int16')
        for irec,(o,e) in enumerate(raw_rec.file_sample_offsets):
            motion = motion_objs[irec]['motion']
            rec = entire_rec.frame_slice(start_frame=o,end_frame=e).astype("float32")
            rec_corrected = interpolate_motion(
                    recording=rec,
                    motion=motion,
                    border_mode="force_extrapolate",
                    spatial_interpolation_method="kriging",
                    sigma_um=30.)

            Parallel(n_jobs = n_jobs)(delayed(_interpolate_rec_parallel)(ori_start = o, new_start = no,new_end = ne,out = out,
                recording_corrected = rec_corrected,
                channels = chan_info.channel_idx.values) 
                for no,ne in tqdm(chunk_indices(np.arange(rec_corrected.get_num_frames())),
                            desc = 'Saving corrected binary file'))    
    return binaryfilepath
    

def dredge_motion_correct_binary_file(binaryfilepath, nchannels, sampling_rate, 
                              channel_coords, 
                              channel_shank, 
                              output_folder,
                              overwrite = True,
                              is_filtered = True,
                              output_dtype = 'int16',#'float32',
                              n_jobs = 10):
    # then do the motion correction using dredge
    # we'll use the dredge implementation from spike interface..
    import spikeinterface.full as si
    global_job_kwargs = dict(n_jobs = n_jobs, chunk_duration = "1s", 
                             pool_engine = "process",
                             max_threads_per_worker = 4)
    si.set_global_job_kwargs(**global_job_kwargs)
    rec = si.read_binary(binaryfilepath, 
                         sampling_frequency=sampling_rate,
                         dtype = np.int16,  # hardcoded for now
                         num_channels = nchannels,
                         is_filtered = is_filtered)
    rec.is_filtered = is_filtered

    rec.set_channel_locations(channel_coords)
    rec.set_channel_groups(channel_shank)

    mo = si.correct_motion(rec, preset='dredge')

    mo.save(folder=pjoin(output_folder,'motion'),
            dtype = output_dtype, 
            format= 'binary')

    # move to filtered_recording
    del rec
    del mo
    del si
    outfile = pjoin(output_folder,'motion','traces_cached_seg0.raw')
    if overwrite:
        import shutil
        shutil.move(outfile,binaryfilepath)
        outfile = binaryfilepath
    # shutil.rmtree(pjoin(foldername,'motion'))        
    return outfile
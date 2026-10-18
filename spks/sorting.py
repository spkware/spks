# Run various spike sorters using spike interface precompiled docker images
# This should allow running it on a fast disk and copying only the relevant files over
# The functions and parameters should be exposed and contained in this module. 
from .utils import *
from .raw import *

def get_probename(filename):
        # get the probe name from a file (only works with spikeglx?)
        probename = re.search(r'\s*imec[0-9]*[a-z]?\s*',str(filename))
        if not probename is None:
                return str(probename.group()).strip('/').strip('.')
        else:
                return 'probe0'

def get_sorting_folder_path(filename,
                            sorting_results_path_rules = ['..','..','{sortname}','{probename}'],
                            sorting_folder_dictionary = dict(sortname = 'sorting',
                                                             probename = 'probe0')):
        '''
        Gets the sorting folder path from a defined rule.

        foldername = get_sorting_folder_path(filename)
        '''
        
        filename = Path(filename)
        if filename.is_file():
                foldername  = filename.parent

        sorting_folder_dictionary['probename'] = get_probename(filename)

        sorting_results_path = foldername #FIXME: not defined if filename isn't file
        for f in sorting_results_path_rules:
                if f == '..':
                        foldername = foldername.parent
                else:
                        foldername = foldername.joinpath(f.format(**sorting_folder_dictionary))
        return foldername

def move_sorting_results(
                scratch_folder,
                original_session_path,
                move_filtered_data = False,
                sorting_results_path_rules = ['..','..','{sortname}','{probename}'],
                sorting_folder_dictionary = dict(
                        sortname = 'sorting',
                        probename = 'probe0')):
        '''
        Move spike sorting results to a standardized folder

        '''
        sorting_results_path = get_sorting_folder_path(
                filename = original_session_path,
                sorting_folder_dictionary=sorting_folder_dictionary,
                sorting_results_path_rules=sorting_results_path_rules)

        files_to_copy = []
        extensions = ['.npy','.tsv','.hdf','.m','.mat','.py','.log']
        if move_filtered_data:
                extensions += ['filtered_recording.*.bin']
        for name in extensions:
                files_to_copy += scratch_folder.glob(f'*{name}')
        if not sorting_results_path.exists():
                sorting_results_path.mkdir(parents=True, exist_ok=True)
        for f in tqdm(files_to_copy,desc = 'Moving files'):
                if os.path.exists(sorting_results_path/f.name):
                        os.remove(sorting_results_path/f.name) #need to delete files before moving to avoid weird permission error
                try:
                        shutil.move(f,sorting_results_path/f.name)
                except Exception as err:
                        print(err)
                        print(f'FAILED to move {f} to {sorting_results_path}')
        return sorting_results_path

def run_kilosort(sessionfiles = [],
                 channels = None,
                 foldername = None,
                 temporary_folder = 'temporary',
                 version = '4.0',
                 sorting_results_path_rules = ['..','..','{sortname}','{probename}'],
                 sorting_folder_dictionary = dict(sortname = None, probename = 'probe0'),
                 do_post_processing = False, 
                 device = 'cuda',
                 gpu_index = 0,
                 split_shanks = False,
                 motion_correction = True,
                 dredge_motion_correction = False,
                 thresholds = None,
                 batch_size = int(60000),
                 lowpass = 300.,
                 highpass = 13000.,
                 filter_pipeline_par =  [dict(function = 'bandpass_filter_gpu',
                                              sampling_rate = 30000,
                                              lowpass = 300,
                                              highpass = 13000,
                                              return_gpu = True),
                                         dict(function = 'phase_shift_gpu',
                                              sample_shifts = None,
                                              return_gpu = True),
                                         dict(function = 'global_car_gpu',
                                              return_gpu = False)]):
        '''
        Runs kilosort given binary files, returns a folder with the results.

        To leave out the channels without signal, add a step to filter_pipeline_par:
            dict(function = 'remove_channels_without_signal', methods = ['amplitude','coherence'])
        This runs before the other steps (see RawRecording.to_binary).
        '''

        using_scratch = False
        if foldername is None:
                foldername = create_temporary_folder(temporary_folder, prefix=f'ks{version}_sorting')
                using_scratch = True
        filter_pipeline_par = [dict(f) for f in filter_pipeline_par] # the default is shared across calls
        for f in filter_pipeline_par:
                if 'lowpass' in f.keys() and not lowpass is None:
                        f['lowpass'] = lowpass
                if 'highpass' in f.keys() and not highpass is None:
                        f['highpass'] = highpass
                
        tt = RawRecording(sessionfiles,device = device,
                          filter_pipeline_par = filter_pipeline_par,
                          return_preprocessed = True)
        dtype = 'int16'
        binaryfilepath = pjoin(foldername,'filtered_recording.{probename}.bin'.format(
                **sorting_folder_dictionary))
        output_folder = os.path.dirname(binaryfilepath)
        
        if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        print(f'Exporting binary to {binaryfilepath}')
        if channels is None:
                channels = tt.channel_info.channel_idx.values
        print(tt.offsets,len(sessionfiles),len(sessionfiles)>1)
        print(sessionfiles)
        binaryfile,metadata = tt.to_binary(binaryfilepath,
                                           filter_pipeline_par = filter_pipeline_par,
                                           channels = channels)

        if dredge_motion_correction:
                from .raw import dredge_motion_correct_binary_file,dredge_motion_correct_across_sessions
                n_jobs = 10 
                print(len(sessionfiles),len(sessionfiles)>1)
                if len(sessionfiles)>1:
                        print('Using multisession dredge.')
                        binaryfilepath = dredge_motion_correct_across_sessions(binaryfilepath=binaryfilepath,
                                                                               metadata = metadata,
                                                                               n_jobs = n_jobs)
                else:
                        print('Using single session dredge.')
                        binaryfilepath = dredge_motion_correct_binary_file(binaryfilepath,
                                                                  nchannels = metadata['nchannels'],
                                                                  sampling_rate = metadata['sampling_rate'],
                                                                  channel_coords = metadata['channel_coords'],
                                                                  channel_shank = metadata['channel_shank'],
                                                                  output_folder = foldername,
                                                                  overwrite = True,
                                                                  output_dtype = 'int16',
                                                                  n_jobs = n_jobs)
                        dtype = 'int16'
                print('Motion corrected done with dredge so skipping kilosort motion.')
                motion_correction = 0

        del tt # RawRecording no longer needed
        channelmappath = pjoin(os.path.dirname(binaryfilepath),'chanMap.mat')
        opspath = pjoin(os.path.dirname(binaryfilepath),'ops.mat')
        if lowpass is None:
                lowpass = 300.
        
        nchannels = metadata['nchannels']
        coords = np.stack(metadata['channel_coords'])
        # make the channelmap file
        chanMap = dict(Nchannels = nchannels,
                       connected = np.ones(nchannels,dtype=bool).T,
                       xcoords = coords[:,0].astype(float),
                       ycoords = coords[:,1].astype(float),
                       chanMap = np.array(metadata['channel_idx'],dtype=np.int64)+1,
                       chanMap0ind = np.array(metadata['channel_idx'],dtype=np.int64),
                       kcoords = np.array(metadata['channel_shank'],dtype=float).T+1,
                       fs = metadata['sampling_rate'])
        from scipy.io import savemat
        savemat(channelmappath, chanMap,appendmat = False)

        if version == '2.5':
                compiled_name = 'kilosort2_5'
                if thresholds is None:
                        thresholds = [9.,3.]
                ops = dict(ops = dict(default_ks25_ops,
                                      NchanTOT=float(metadata['nchannels']),
                                      Nchan = float(len(metadata['channel_idx'])),
                                      fbinary = binaryfilepath,
                                      fproc = pjoin(output_folder,'temp_wh.dat'),
                                      chanMap = channelmappath,
                                      fs = metadata['sampling_rate'],
                                      doCorrection = int(motion_correction),
                                      fshigh = lowpass,
                                      Th = thresholds,
                                      GPU = gpu_index + 1)) # indices are one based ...
                matlabcommand = kilosort25_matlabcommand
        elif version == '3.0':
                compiled_name = 'kilosort3_0'
                if thresholds is None:
                        thresholds = [9.,9.]
                ops = dict(ops = dict(default_ks30_ops,
                                      NchanTOT=float(metadata['nchannels']),
                                      Nchan = float(len(metadata['channel_idx'])),
                                      fbinary = binaryfilepath,
                                      fproc = pjoin(output_folder,'temp_wh.dat'),
                                      chanMap = channelmappath,
                                      fs = metadata['sampling_rate'],
                                      doCorrection = int(motion_correction),
                                      fshigh = lowpass,
                                      Th = thresholds,
                                      GPU = gpu_index + 1)) # indices are one based ...
                matlabcommand = kilosort30_matlabcommand
        if version in ['2.5','3.0']:
                # save the files 
                savemat(opspath, ops,appendmat = False)
                import shutil
                if not shutil.which(compiled_name) is None:
                        os.system(f'{compiled_name} {output_folder}') # easier to kill than subprocess?
                else:  # just run using a local installation..
                        matlabfile = pjoin(output_folder,'run_ks.m')
                        with open(matlabfile,'w') as f:
                                f.write(matlabcommand.format(output_folder = output_folder))
                        cmd = """matlab -nodisplay -nosplash -r "run('{0}');" """.format(matlabfile)
                        os.system(cmd) # easier to kill than subprocess?
                driftfile = pjoin(output_folder,'drift.mat')
                if os.path.exists(driftfile):
                        from scipy.io import loadmat
                        drift = loadmat(driftfile)
                        if 'Wrot' in drift.keys():
                                # kilosort sorted drift corrected data, unwhiten it into the binary so the
                                # waveforms come from corrected data
                                kilosort_unwhiten_binary(ops['ops']['fproc'], binaryfilepath, nchannels, drift['Wrot'])
        elif version == '4.0':
                if thresholds is None:
                        thresholds = [9.,8.]
                res = run_kilosort4(
                        device = device,
                        foldername = foldername,
                        binaryfilepath = binaryfilepath,
                        metadata = metadata,
                        motion_correction = motion_correction,
                        thresholds = thresholds,
                        dtype = dtype,
                        batch_size = batch_size)
        else:
                raise(OSError('Undefined version {version}'))
        if do_post_processing:
                foldername = kilosort_post_processing(
                        foldername,
                        sessionfiles,
                        move =  using_scratch, 
                        binaryfilepath = binaryfilepath,
                        sorting_results_path_rules = sorting_results_path_rules,
                        sorting_folder_dictionary = sorting_folder_dictionary)
        return foldername

def run_kilosort4(device, foldername, binaryfilepath, 
                  metadata, 
                  dtype = 'int16',
                  thresholds = None, # pass a list if needed
                  do_car = False,
                  batch_size = int(60000), 
                  motion_correction = 1):
        nchannels = metadata['nchannels']
        coords = np.stack(metadata['channel_coords'])
        
        yc = coords[:,1].astype(np.float32)
        xc = coords[:,0].astype(np.float32)
        fix_shanks = False # flag to fix the phy coords # this is now fixed on v4.04
        if fix_shanks:
                # lets stack the shanks... because kilosort 4.0 can not handle multiple shanks.. 
                previous = 0
                for shank in np.unique(metadata['channel_shank']):
                        idx = np.where(metadata['channel_shank'] == shank)
                        offset = np.max(yc[idx])
                        yc[idx] = (yc[idx]-np.min(yc[idx])) + previous
                        xc[idx] = (xc[idx]-np.min(xc[idx]))
                        previous += 100 + offset

        probe = dict(n_chan = nchannels,
                     xc = xc,
                     yc = yc,
                     chanMap = np.array(metadata['channel_idx'], dtype=int),
                     kcoords = np.array(metadata['channel_shank'], dtype=int).T)

        from kilosort import run_kilosort
        settings = dict(fs = metadata['sampling_rate'],
                        n_chan_bin = nchannels,
                        batch_size = batch_size,
                        data_dir = foldername)
        
        settings['nblocks'] = int(motion_correction)
        if not thresholds is None:
                settings['Th_universal'] = thresholds[0]
                settings['Th_learned'] = thresholds[1]

        res  = run_kilosort(filename = binaryfilepath,
                            results_dir = foldername,
                            settings = settings, 
                            data_dtype = dtype,
                            probe = probe,
                            device = device,
                            do_CAR = do_car,
                            save_preprocessed_copy = False,
                            save_extra_vars = True) # save pc_features
        
        if fix_shanks:
                ops = res[0]
                st = res[1]
                tF = res[3]
                ops['xc'],ops['yc'] = coords.astype(np.float32).T # recompute the spike positions
                from kilosort.postprocessing import compute_spike_positions
                positions = compute_spike_positions(st,tF,ops)
                np.save(Path(foldername)/'spike_positions.npy',np.vstack(positions).T) # overwrite spike positions
                np.save(Path(foldername)/'channel_positions.npy',coords.astype(float)) # save the correct channel positions
        ops = res[0]
        if ops.get('dshift', None) is not None:
                # kilosort sorted drift corrected data, correct the binary the same way so the
                # waveforms come from corrected data
                kilosort_drift_correct_binary(binaryfilepath, ops, nchannels, dtype = dtype, device = device)
        del res, ops
        free_gpu()
        return True

def kilosort_unwhiten_binary(whitened_file, binaryfilepath, nchannels, whitening_matrix,
                             dtype = 'int16', n_jobs = 4, chunksize = 60000):
        '''Write the Kilosort 2.5/3.0 processed data, unwhitened, over a binary file.

        ``whitened_file`` (``temp_wh.dat``) holds the data Kilosort sorted:
        filtered, multiplied by ``rez.Wrot`` (whitening scaled by
        ``ops.scaleproc``) and drift corrected.  Multiplying by the inverse of
        ``rez.Wrot`` brings it back to the units of the binary file.  Kilosort
        shifts the whitened data, so the result is the drift correction seen
        through the whitening, close to the correction of the original data
        when the whitening is local.  ``n_jobs`` joblib threads write chunks of
        ``chunksize`` samples at the same time.
        '''
        from joblib import Parallel, delayed
        dtype = np.dtype(dtype)
        nsamples = int(os.path.getsize(binaryfilepath)//(nchannels*dtype.itemsize))
        nwhitened = int(os.path.getsize(whitened_file)//(nchannels*dtype.itemsize))
        if nwhitened < nsamples:
                raise ValueError(f'{whitened_file} has {nwhitened} samples, {binaryfilepath} has {nsamples}.')
        dat = np.memmap(binaryfilepath, dtype = dtype, mode = 'r+', shape = (nsamples, nchannels))
        whitened = np.memmap(whitened_file, dtype = dtype, mode = 'r', shape = (nwhitened, nchannels))
        # kilosort whitens rows of samples: whitened = data @ Wrot
        unwhitening = np.linalg.inv(np.asarray(whitening_matrix, dtype = np.float64)).astype(np.float32)
        limits = np.iinfo(dtype)
        def unwhiten(start):
                end = min(start + chunksize, nsamples)
                X = np.asarray(whitened[start:end], dtype = np.float32) @ unwhitening
                dat[start:end] = np.clip(np.round(X), limits.min, limits.max).astype(dtype)
        Parallel(n_jobs = n_jobs, backend = 'threading')(
                delayed(unwhiten)(start) for start in tqdm(range(0, nsamples, chunksize),
                                                           desc = 'Unwhitening the kilosort data into the binary file'))
        dat.flush()
        del dat, whitened

def kilosort_drift_correct_binary(binaryfilepath, ops, nchannels, dtype = 'int16', device = 'cuda', n_jobs = 4):
        '''Apply the Kilosort 4 drift correction to a binary file, in place.

        Each batch of ``ops['batch_size']`` samples is multiplied by the drift
        matrix of that batch (``kilosort.preprocessing.get_drift_matrix``) on the
        channels of the probe channel map, as Kilosort does before sorting.  The
        whitening is left out, so the data keep their units.  ``n_jobs`` joblib
        threads correct batches at the same time; each batch writes only its own samples.
        '''
        import torch
        from kilosort.preprocessing import get_drift_matrix
        dtype = np.dtype(dtype)
        nsamples = int(os.path.getsize(binaryfilepath)//(nchannels*dtype.itemsize))
        dat = np.memmap(binaryfilepath, dtype = dtype, mode = 'r+', shape = (nsamples, nchannels))
        chanmap = np.asarray(ops['probe']['chanMap'], dtype = int)
        if np.array_equal(chanmap, np.arange(nchannels)):
                chanmap = slice(None) # all channels, read and write without copying the columns
        batch_size = int(ops['batch_size'])
        first_sample = int(ops['settings'].get('tmin', 0)*ops['fs'])
        limits = np.iinfo(dtype)
        def correct_batch(ibatch):
                start = first_sample + ibatch*batch_size
                end = min(start + batch_size, nsamples)
                if start >= end:
                        return
                M = get_drift_matrix(ops, ops['dshift'][ibatch], device = device)
                X = torch.from_numpy(np.asarray(dat[start:end, chanmap], dtype = np.float32)).to(device)
                X = (M.to(X.dtype) @ X.T).T
                dat[start:end, chanmap] = np.clip(np.round(X.cpu().numpy()), limits.min, limits.max).astype(dtype)
        # threads share the memmap and the device
        from joblib import Parallel, delayed
        Parallel(n_jobs = n_jobs, backend = 'threading')(
                delayed(correct_batch)(ibatch) for ibatch in tqdm(range(len(ops['dshift'])),
                                                                  desc = 'Drift correcting the binary file'))
        dat.flush()
        del dat

def kilosort_post_processing(resultsfolder,
                             sessionfolder,
                             move = False,
                             binaryfilepath = None,
                             sorting_results_path_rules = ['..','..','{sortname}','{probename}'],
                             sorting_folder_dictionary = dict(
                                     sortname = 'kilosort',
                                     probename = 'probe0'),
                             max_n_spikes = 1000):
        '''
        Post processing for kilosort results

        1. remove duplicates
        2. compute_waveforms
        3. store a sample of 1000 waveforms to disk
        4. move the files to a new folder and if so delete the scratch folder

        '''
        # 1. remove duplicates
        from .clusters import Clusters
        resultsfolder = Path(resultsfolder)
        sp = Clusters(resultsfolder,get_metrics = False,get_waveforms=False,load_template_features = True)
        sp.remove_duplicate_spikes(overwrite_phy=True)
        del sp
        # 2. compute_waveforms and store to disk
        sp = Clusters(resultsfolder,get_metrics = False,get_waveforms=False)
        meta = load_dict_from_h5(list(resultsfolder.glob('filtered_recording.*.metadata.hdf'))[0])
        from .io import map_binary
        if binaryfilepath is None:
                binaryfilepath = list(resultsfolder.glob('filtered_recording.*.bin'))[0]
        data = map_binary(binaryfilepath,meta['nchannels'])
        # don't filter the waveforms because it was done before.
        sp.extract_waveforms(data,np.arange(meta['nchannels']),
                             max_n_spikes=max_n_spikes,
                             save_folder_path = resultsfolder,filter_par = None)
        del sp
        # Compute metrics and mean waveforms
        sp = Clusters(resultsfolder,get_metrics = True, get_waveforms=True, load_template_features = True)
        del sp # close so it can move
        # 4. move the files to a new folder
        if move:
                if type(sessionfolder) in [list]:
                        folder = sessionfolder[0]
                        print(f'Saving to {folder}')
                else:
                        folder = sessionfolder
                outputfolder = move_sorting_results(
                        resultsfolder,
                        folder,
                        sorting_results_path_rules = sorting_results_path_rules,
                        sorting_folder_dictionary = sorting_folder_dictionary)
                # 5. delete the scratch
                shutil.rmtree(resultsfolder)
                resultsfolder = outputfolder
        return resultsfolder # Clusters(resultsfolder) to open

from .io import list_spikeglx_binary_paths

def sort_multiprobe_sessions(sessions,
                             temporary_folder = '/scratch',
                             method = 'kilosort2.5',
                             sorting_results_path_rules = ['..','..','{sortname}','{probename}'],
                             sorting_folder_dictionary = dict(
                                     sortname = 'kilosort', probename = 'probe0'),
                             do_post_processing = True,
                             move = True,
                             device = 'cuda',
                             gpu_index = 0,
                             motion_correction = True):
        '''
        Sort multiprobe neuropixels recordings (will concatenate multiple sessions if a list is passed).
        '''
        if not type(sessions) is list:
                sessions = [sessions]

        tmp = [list_spikeglx_binary_paths(s) for s in sessions]
        all_probe_dirs = []
        for iprobe in range(len(tmp[0])):
                all_probe_dirs.append([t[iprobe][0] for t in tmp])
        results = []
        for probepath in all_probe_dirs:
                print('Running {1} on sessions {0}'.format(' ,'.join(probepath),method))
                sorting_folder_dictionary['probename'] = get_probename(probepath)
                
                results_folder = run_kilosort(sessionfiles = probepath,
                                              version = method.strip('kilosort'),
                                              temporary_folder = temporary_folder,
                                              sorting_results_path_rules = sorting_results_path_rules,
                                              sorting_folder_dictionary = sorting_folder_dictionary,
                                              do_post_processing = False,
                                              motion_correction = motion_correction,
                                              device=device, gpu_index = gpu_index)
                print('Completed {1} Results folder: {0}'.format(results_folder,method))
                if do_post_processing:
                        results_folder = kilosort_post_processing(
                                results_folder,
                                probepath, 
                                sorting_results_path_rules = sorting_results_path_rules,
                                sorting_folder_dictionary = sorting_folder_dictionary,
                                move = move)
                        print('Completed sorting for results folder: {0}'.format(results_folder))
                results.append(results_folder)
        return results

kilosort25_matlabcommand = '''
load(fullfile('{output_folder}','ops.mat'))
load(fullfile('{output_folder}','chanMap.mat'))
% This assumes kilosort 2.5 and all are installed and on the path
% preprocess data to create temp_wh.dat
rez = preprocessDataSub(ops);
rez = datashift2(rez, ops.doCorrection); % last input is for shifting data
% keep the drift estimate that was applied (yblk is only there when datashift2 stores it)
% and the whitening matrix, to unwhiten the drift corrected temp_wh.dat
if ops.doCorrection && isfield(rez,'dshift')
    dshift = rez.dshift; NT = rez.ops.NT; Wrot = rez.Wrot; yblk = [];
    if isfield(rez.ops,'yblk'), yblk = rez.ops.yblk; end
    save(fullfile('{output_folder}','drift.mat'),'dshift','yblk','NT','Wrot');
end

% ORDER OF BATCHES IS NOW RANDOM, controlled by random number generator
iseed = 1;                 
rez = learnAndSolve8b(rez, iseed); % main tracking and template matching algorithm

% OPTIONAL: remove double-counted spikes - solves issue in which individual spikes are assigned to multiple templates.
% See issue 29: https://github.com/MouseLand/Kilosort/issues/29
% rez = remove_ks2_duplicate_spikes(rez);

rez = find_merges(rez, 1);       % final merges
rez = splitAllClusters(rez, 1);  % final splits by SVD
rez = set_cutoff(rez);           % decide on cutoff
rez.good = get_good_units(rez);  % eliminate widely spread waveforms (likely noise)
fprintf('found %d good units', sum(rez.good>0))
fprintf('Saving results to Phy  ')
rezToPhy(rez, '{output_folder}');            % write to Phy
exit(1);
'''

kilosort30_matlabcommand = '''
load(fullfile('{output_folder}','ops.mat'))
load(fullfile('{output_folder}','chanMap.mat'))
% This assumes kilosort 3.0 and all are installed and on the path
% preprocess data to create temp_wh.dat
rez = preprocessDataSub(ops);
rez = datashift2(rez, ops.doCorrection); % last input is for shifting data
% keep the drift estimate that was applied (yblk is only there when datashift2 stores it)
% and the whitening matrix, to unwhiten the drift corrected temp_wh.dat
if ops.doCorrection && isfield(rez,'dshift')
    dshift = rez.dshift; NT = rez.ops.NT; Wrot = rez.Wrot; yblk = [];
    if isfield(rez.ops,'yblk'), yblk = rez.ops.yblk; end
    save(fullfile('{output_folder}','drift.mat'),'dshift','yblk','NT','Wrot');
end

[rez, st3, tF] = extract_spikes(rez);
rez = template_learning(rez, tF, st3);
[rez, st3, tF] = trackAndSort(rez);
rez = final_clustering(rez, tF, st3);
rez = find_merges(rez, 1);

fprintf('Saving results to Phy  ')
rezToPhy2(rez, '{output_folder}');           % write to Phy
exit(1);
'''

default_ks25_ops = dict(
        datatype = 'dat',
        trange = [0.,np.inf],
        CAR = 1.,
        nblocks = 5.,
        sig = 20.,
        lam = 10.,
        AUCsplit = 0.9,
        minFR = 1./50,
        momentum = [20.,400.],
        sigmaMask = 30.,
        ThPre  = 8.,
        spkTh = -6.,
        reorder = 1.,
        nskip = 25.,
        nfilt_factor = 4.,
        ntbuff  = 64.,
        NT = 65600.,
        whiteningRange = 32.,
        nSkipCov = 25.,
        scaleproc = 200.,
        nPCs = 3,
        useRam = 0,
        nt0 = 61.)

default_ks30_ops = dict(
        datatype = 'dat',
        trange = [0.,np.inf],
        CAR = 1.,
        nblocks = 5.,
        sig = 20.,
        lam = 20.,
        AUCsplit = 0.8,
        minFR = 1./50,
        momentum = [20.,400.],
        sigmaMask = 30.,
        ThPre  = 8.,
        spkTh = -6.,
        reorder = 1.,
        nskip = 25.,
        nfilt_factor = 4.,
        ntbuff  = 64.,
        NT = 65600.,
        whiteningRange = 32.,
        nSkipCov = 25.,
        scaleproc = 200.,
        nPCs = 3,
        useRam = 0,
        nt0 = 61.)
                        

class SpikeSorting(object):
        def __init__(raw_files, output_folder,
                     filter_pipeline_par = [dict(function = 'bandpass_filter_gpu',
                                                 sampling_rate = 30000,
                                                 lowpass = 300,
                                                 highpass = 10000,
                                                 return_gpu = False),
                                            dict(function = 'global_car_gpu',
                                                 return_gpu = True)],
                     temporary_folder = None, motion_correction = True, **kwargs):
                ''' Run a spike sorter. 
        1) Creates the output folder.
        2) Concatenates the input files.
        3) Writes a json file with the onsets and offsets of each file, the channelmap
        4) Downloads a sorter image and runs it in the temporaty folder
        5) Copies the files to the output folder and cleans the temporary folder


    THIS IS A PLACEHOLDER FOR NOW.
        '''
                pass



        

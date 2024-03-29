# Run various spike sorters using spike interface precompiled docker images
# This should allow running it on a fast disk and copying only the relevant files over
# The functions and parameters should be exposed and contained in this module. 
from .utils import *
from .raw import *

def get_probename(filename):
        # get the probe name from a file (only works with spikeglx?)
        probename = re.search('\s*imec[0-9]*[a-z]?\s*',str(filename))
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
                shutil.move(f,sorting_results_path/f.name)
        return sorting_results_path

def run_kilosort(sessionfiles = [],
                 foldername = None,
                 temporary_folder = 'temporary',
                 version = '2.5',
                 sorting_results_path_rules = ['..','..','{sortname}','{probename}'],
                 sorting_folder_dictionary = dict(sortname = None, probename = 'probe0'),
                 do_post_processing = False, device = 'cuda',gpu_index = 0,
                 motion_correction = True,
                 thresholds = None,
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
        '''
        
        using_scratch = False
        if foldername is None:
                foldername = create_temporary_folder(temporary_folder, prefix=f'ks{version}_sorting')
                using_scratch = True
                
        for f in filter_pipeline_par:
                if 'lowpass' in f.keys() and not lowpass is None:
                        f['lowpass'] = lowpass
                if 'highpass' in f.keys() and not highpass is None:
                        f['highpass'] = highpass
                        
        tt = RawRecording(sessionfiles,device = device,
                          filter_pipeline_par = filter_pipeline_par,
                          return_preprocessed = True)
        binaryfilepath = pjoin(foldername,'filtered_recording.{probename}.bin').format(
                **sorting_folder_dictionary)
        output_folder = os.path.dirname(binaryfilepath)
        
        if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        print(f'Exporting binary to {binaryfilepath}')
        binaryfile,metadata = tt.to_binary(binaryfilepath,
                                           filter_pipeline_par = filter_pipeline_par,
                                           channels = tt.channel_info.channel_idx.values)
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
                ops = dict(dict(default_ks25_ops,
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
                ops = dict(dict(default_ks30_ops,
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
        elif version == '4.0':
                ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate = run_kilosort4(
                        device,
                        foldername,
                        binaryfilepath,
                        binary,
                        metadata)
        else:
                raise(OSError('Undefined version {version}'))
        if do_post_processing:
                foldername = kilosort_post_processing(
                        foldername,
                        sessionfiles,
                        move =  using_scratch, 
                        sorting_results_path_rules = sorting_results_path_rules,
                        sorting_folder_dictionary = sorting_folder_dictionary)
        return foldername

def run_kilosort4(device, foldername, binaryfilepath, binary, metadata):
        nchannels = metadata['nchannels']
        coords = np.stack(metadata['channel_coords'])

        # lets stack the shanks... because kilosort 4.0 can not handle multiple shanks.. 
        fix_shanks = True # flag to fix the phy coords
        yc = coords[:,1].astype(float)
        xc = coords[:,0].astype(float)
        previous = 0
        for shank in np.unique(metadata['channel_shank']):
                idx = np.where(metadata['channel_shank']==shank)
                offset = np.max(yc[idx])
                yc[idx] = (yc[idx]-np.min(yc[idx])) + previous
                xc[idx] = (xc[idx]-np.min(xc[idx]))
                previous += 100 + offset # stack the channels on top of each other...

        probe = dict(n_chan = nchannels,
                     xc = xc,
                     yc = yc,
                     chanMap = np.array(metadata['channel_idx'],dtype=int),
                     kcoords = np.array(metadata['channel_shank'],dtype=int).T)

        from kilosort import run_kilosort
        settings = dict(fs = metadata['sampling_rate'],
                        n_chan_bin = nchannels,
                        data_dir = foldername)
        if not motion_correction:
                settings['nblocks'] = 0

        ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate = run_kilosort(filename = binaryfilepath,
                                                                                          results_dir = foldername,
                                                                                          settings=settings, 
                                                                                          data_dtype = 'int16', # hardcoded now..
                                                                                          probe = probe,
                                                                                          device = device)
        return ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate

if 'fix_shanks' in dir():
    yc = coords[:,1].astype(float)
    xc = coords[:,0].astype(float)



def kilosort_post_processing(resultsfolder,
                             sessionfolder,
                             move = False,
                             sorting_results_path_rules = ['..','..','{sortname}','{probename}'],
                             sorting_folder_dictionary = dict(
                                     sortname = 'kilosort25',
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
        data = map_binary(list(resultsfolder.glob('filtered_recording.*.bin'))[0],meta['nchannels'])
        # don't filter the waveforms because it was done before.
        sp.extract_waveforms(data,np.arange(meta['nchannels']),
                             max_n_spikes=max_n_spikes,
                             save_folder_path = resultsfolder,filter_par = None)
        del sp
        # Compute metrics and mean waveforms
        sp = Clusters(resultsfolder,get_metrics = True, get_waveforms=True, load_template_features = True)
        
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
        return resultsfolder

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



        

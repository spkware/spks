# Run various spike sorters using spike interface precompiled docker images
# This should allow running it on a fast disk and copying only the relevant files over
# The functions and parameters should be exposed and contained in this module. 


from .utils import *
from .raw import *

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
        # get the probename from a filename
        probename = re.search('\s*imec(\d)\s*',str(filename))
        if not probename is None:
                sorting_folder_dictionary['probename'] = probename.group()

        sorting_results_path = foldername
        for f in sorting_results_path_rules:
                if f == '..':
                        foldername = foldername.parent
                else:
                        foldername = foldername.joinpath(f.format(**sorting_folder_dictionary))
        return foldername

def move_sorting_results(scratch_folder, original_session_path,
                         sorting_results_path_rules = ['..','..','{sortname}','{probename}'],
                         sorting_folder_dictionary = dict(sortname = 'sorting',
                                                          probename = 'probe0')):
        '''
        Move spike sorting results to a standardized folder

        '''
        sorting_results_path = get_sorting_folder_path(
                filename = original_session_path,
                sorting_folder_dictionary=sorting_folder_dictionary,
                sorting_results_path_rules=sorting_results_path_rules)

        files_to_copy = []
        for name in ['.npy','.tsv','.hdf','.m','.mat','.py','.log','filtered_recording.*.bin']:
                files_to_copy += scratch_folder.glob(f'*{name}')
        if not sorting_results_path.exists():
                sorting_results_path.mkdir(parents=True, exist_ok=True)
        for f in tqdm(files_to_copy,desc = 'Moving files'):
                shutil.move(f,sorting_results_path)

        return sorting_results_path

def get_si_docker_sorter(sorter = 'kilosort2_5-compiled-base'):
        '''
        Pulls docker image; right now supports only kilosort25; add support for other sorters is TODO
        '''
        import docker
        client = docker.from_env()
        return client.images.pull('spikeinterface/'+sorter)

class SpikeSorting(object):
        def __init__(raw_files, output_folder,
                     preprocessing = [lambda x: bandpass_filter_gpu(x,30000,300,5000),
                                      lambda x: global_car_gpu(x,return_gpu=False)],
                     temporary_folder = None, **kwargs):
                ''' Run a spike sorter. 
        1) Creates the output folder.
        2) Concatenates the input files.
        3) Writes a json file with the onsets and offsets of each file, the channelmap
        4) Downloads a sorter image and runs it in the temporaty folder
        5) Copies the files to the output folder and cleans the temporary folder


    THIS IS A PLACEHOLDER FOR NOW.
        '''
                pass

def ks25_run(sessionfiles = [],
             foldername = None,
             temporary_folder = '/scratch',
             use_docker = False,
             sorting_results_path_rules = ['..','..','{sortname}','{probename}'],
             sorting_folder_dictionary = dict(sortname = 'kilosort25', probename = 'probe0'),
             do_post_processing = False, device = 'cuda',gpu_index = 0,
             use_precompiled = False):
        using_scratch = False
        if foldername is None:
                foldername = create_temporary_folder(temporary_folder, prefix='ks25_sorting')
                using_scratch = True
        tt = RawRecording(sessionfiles,device = device)
        binaryfilepath = pjoin(foldername,'filtered_recording.{probename}.bin').format(
                **sorting_folder_dictionary)
        output_folder = os.path.dirname(binaryfilepath)

        if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        binaryfile,metadata = tt.to_binary(binaryfilepath,
                                           channels = tt.channel_info.channel_idx.values)
        free_gpu()
        channelmappath = pjoin(os.path.dirname(binaryfilepath),'chanMap.mat')
        opspath = pjoin(os.path.dirname(binaryfilepath),'ops.mat')
        # kilosort options TODO: expose the options
        ops = dict(ops=dict(NchanTOT=float(metadata['nchannels']),
                            Nchan = float(len(metadata['channel_idx'])),
                            datatype = 'dat',
                            fbinary = binaryfilepath,
                            fproc = pjoin(output_folder,'temp_wh.dat'),
                            trange = [0.,np.inf],
                            chanMap = channelmappath,
                            fs = metadata['sampling_rate'],
                            CAR = 1.,
                            fshigh = 150.,
                            nblocks = 5.,
                            sig = 20.,
                            Th = [9.,3.],
                            lam = 10.,
                            AUCsplit = 0.9,
                            minFR = 1./50,
                            momentum = [20.,400.],
                            sigmaMask = 30.,
                            ThPre  = 8.,
                            spkTh = -6.,
                            reorder = 1.,
                            nskip = 25.,
                            GPU = gpu_index + 1, # indices are one based ...
                            nfilt_factor = 4.,
                            ntbuff  = 64.,
                            NT = 65600.,
                            whiteningRange = 32.,
                            nSkipCov = 25.,
                            scaleproc = 200.,
                            nPCs = 3,
                            useRam = 0,
                            doCorrection = 1,
                            nt0 = 61.))
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
        # save the files 
        from scipy.io import savemat
        savemat(opspath, ops,appendmat = False)
        savemat(channelmappath, chanMap,appendmat = False)
        if use_precompiled:
                os.system(f'kilosort2_5 {output_folder}') # easier to kill than subprocess?
        elif use_docker:
                image = get_si_docker_sorter('kilosort2_5-compiled-base')
                client  = image.client
                import docker
                container = client.containers.run(
                        image,
                        command=f'ks2_5_compiled {output_folder}',
                        volumes=['{0}:{0}'.format(p)],
                        device_requests = [
                                docker.types.DeviceRequest(count=-1,capabilities=[["gpu"]])],
                        detach=True)
                for l in container.logs(stream=True,tail=True):
                        print('  [kilosort] - {0}'.format(l.decode().strip('\n')))
        else:
                # just run using a local installation..
                kilosort25_file = '''
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
                matlabfile = pjoin(output_folder,'run_ks.m')
                with open(matlabfile,'w') as f:
                        f.write(kilosort25_file.format(output_folder = output_folder))
                cmd = """matlab -nodisplay -nosplash -r "run('{0}');" """.format(matlabfile)
                os.system(cmd) # easier to kill than subprocess?
        if do_post_processing:
                foldername = ks25_post_processing(
                        foldername,
                        sessionfiles,
                        move =  using_scratch, 
                        sorting_results_path_rules = sorting_results_path_rules,
                        sorting_folder_dictionary = sorting_folder_dictionary)
        return foldername

def ks25_post_processing(resultsfolder,
                         sessionfolder,
                         move = False,
                         sorting_results_path_rules = ['..','..','{sortname}','{probename}'],
                         sorting_folder_dictionary = dict(
                                 sortname = 'kilosort25',
                                 probename = 'probe0'),
                         max_n_spikes = 500):
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
        sp = Clusters(resultsfolder,get_metrics = False)
        sp.remove_duplicate_spikes(overwrite_phy=True)
        # 2. compute_waveforms and store to disk        
        meta = load_dict_from_h5(list(resultsfolder.glob('filtered_recording.*.metadata.hdf'))[0])
        from .io import map_binary
        data = map_binary(list(resultsfolder.glob('filtered_recording.*.bin'))[0],meta['nchannels'])
        sp.compute_waveforms(data,chmap,max_n_spikes,save_waveforms = True)
        
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

def ks25_sort_multiprobe_sessions(sessions,
                                  temporary_folder = '/scratch', 
                                  sorting_results_path_rules = ['..','..','{sortname}','{probename}'],
                                  sorting_folder_dictionary = dict(
                                          sortname = 'kilosort25', probename = 'probe0'),
                                  do_post_processing = True,
                                  move = True,
                                  device = 'cuda',
                                  gpu_index = 0,
                                  use_precompiled = False,
                                  use_docker = False):
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
                print('Running KILOSORT 2.5 on sessions {0}'.format(' ,'.join(probepath)))
                results_folder = ks25_run(sessionfiles = probepath,
                                          temporary_folder = temporary_folder,
                                          sorting_results_path_rules = sorting_results_path_rules,
                                          sorting_folder_dictionary = sorting_folder_dictionary,
                                          do_post_processing = False,
                                          use_docker = use_docker,
                                          use_precompiled = use_precompiled,
                                          device=device, gpu_index = gpu_index)
                print('Completed KILOSORT 2.5. Results folder: {0}'.format(results_folder))
                if do_post_processing:
                        results_folder = ks25_post_processing(
                                results_folder,
                                probepath, 
                                sorting_results_path_rules = sorting_results_path_rules,
                                sorting_folder_dictionary = sorting_folder_dictionary,
                                move = move)
                        print('Completed sorting for results folder: {0}'.format(results_folder))
                results.append(results_folder)
        return results


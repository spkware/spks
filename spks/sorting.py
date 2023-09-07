# Run various spike sorters using spike interface precompiled docker images
# This should allow running it on a fast disk and copying only the relevant files over
# The functions and parameters should be exposed and contained in this module. 

try:
    import docker
except:
    print('Could not load docker. Try pip install docker.')

from .utils import *
from .raw import *

def get_si_docker_sorter(sorter = 'kilosort2_5-compiled-base'):
    '''
    Pulls docker image; right now supports only kilosort25; add support for other sorters is TODO
    '''
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



def run_ks25(sessionfiles = [], foldername = None, temporary_folder = '/scratch', use_docker = False):
        if foldername is None:
                foldername = create_temporary_folder(temporary_folder, prefix='ks25_sorting')
        tt = RawRecording(sessionfiles)
        binaryfilepath = pjoin(foldername,'filtered_recording.ap.bin')
        if not os.path.exists(os.path.dirname(binaryfilepath)):
                os.makedirs(os.path.dirname(binaryfilepath))
        binaryfile,metadata = tt.to_binary(binaryfilepath, channels = tt.channel_info.channel_idx.values)
        free_gpu()
        channelmappath = pjoin(os.path.dirname(binaryfilepath),'chanMap.mat')
        opspath = pjoin(os.path.dirname(binaryfilepath),'ops.mat')
        # kilosort options TODO: expose the options
        ops = dict(ops=dict(NchanTOT=float(metadata['nchannels']),
                Nchan = float(len(metadata['channel_idx'])),
                datatype = 'dat',
                fbinary = binaryfilepath,
                fproc = pjoin(os.path.dirname(binaryfilepath),'temp_wh.dat'),
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
                GPU = 1,
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
        chanMap = dict(Nchannels = nchannels,
                connected = np.ones(nchannels,dtype=bool).T,xcoords = coords[:,0].astype(float),ycoords = coords[:,1].astype(float),
                chanMap = np.array(metadata['channel_idx'],dtype=np.int64)+1,
                chanMap0ind = np.array(metadata['channel_idx'],dtype=np.int64),
                kcoords = np.array(metadata['channel_shank'],dtype=float).T+1, fs = metadata['sampling_rate'])
        # save the files 
        from scipy.io import savemat
        savemat(opspath, ops,appendmat = False)
        savemat(channelmappath, chanMap,appendmat = False)
        # get the kilosort image and run it.
        if use_docker:
                image = get_si_docker_sorter('kilosort2_5-compiled-base')
                client  = image.client
                import docker
                p = os.path.dirname(binaryfilepath)
                container = client.containers.run(image,  command='ks2_5_compiled {0}'.format(p),
                                        volumes=['{0}:{0}'.format(p)],
                                        device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],detach=True)
                for l in container.logs(stream=True,tail=True):
                        print('  [kilosort] - {0}'.format(l.decode().strip('\n')))
        else:
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
                matlabfile = pjoin(os.path.dirname(binaryfilepath),'run_ks.m')
                with open(matlabfile,'w') as f:
                        f.write(kilosort25_file.format(output_folder = os.path.dirname(binaryfilepath)))
                cmd = """matlab -nodisplay -nosplash -r "run('{0}');" """.format(matlabfile)
                os.system(cmd) # easier to kill than subprocess?
        return foldername
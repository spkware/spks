from .utils import *
import tqdm as tqdm
from pathlib import Path

__KILOSORT_CLUSTERS_FILENAME = 'cluster_KSLabel.tsv'
__PHY_CLUSTERS_FILENAME = 'cluster_group.tsv'

def is_phy_curated(*sortfolders):
    """takes arbitrary number of spike sorting result folders (need to unpack list when calling) and will return a list
    of wether those results have been curated in phy or not. Will also accept a single string and return a boolean."""
    #TODO: verify this works when curation has been done
    is_curated = []
    for folder in sortfolders:
        ksdata = pd.read_csv(Path(folder) / __KILOSORT_CLUSTERS_FILENAME, sep='\t',header=0)
        phydata = pd.read_csv(Path(folder) / __PHY_CLUSTERS_FILENAME, sep='\t', header=0)
        is_curated.append(not ksdata.equals(phydata))
    if len(is_curated) == 1:
        is_curated = is_curated[0]
    return is_curated

def list_spikeglx_binary_paths(subject_dir):
    """return a list of spikeglx files present for each probe"""
    #TODO: check for missing probe data
    #TODO: add is_sorted flag to only grab sorted sessions
    bin_paths = list(Path(subject_dir).expanduser().glob('**/*.ap.bin'))
    
    # the probe name is the end of the file 'imecX'
    probe_names = natsorted(set([path.name for path in bin_paths])) #assumes the last folder in the path defines the probe name
    import re
    probe = re.search('\s*imec(\d)\s*',p).group()

    all_probe_dirs = []
    for probe_name in probe_names:
        probe_dirs = [str(folder) for folder in bin_paths if folder.name == probe_name]
        probe_dirs = natsorted(probe_dirs, key=str)
        all_probe_dirs.append(probe_dirs)
    return all_probe_dirs

def list_kilosort_result_paths(subject_dir, return_dates=False):
    """return a list of spike-sorting paths for each probe"""
    #TODO: add has_phy flag to only grab curated sessions
    #TODO: check for missing probe data
    ks_dirs = Path(subject_dir).expanduser().glob('**/spike_times.npy')
    ks_dirs = [folder.parent for folder in ks_dirs]
    #if has_phy:
    #    ks_dirs = [folder for folder in ks_dirs if (folder/'.phy').is_dir()]
    probe_names = natsorted(set([path.name for path in ks_dirs])) #assumes the last folder in the path defines the probe name
    all_probe_dirs = []
    for probe_name in probe_names:
        probe_dirs = [str(folder) for folder in ks_dirs if folder.name == probe_name]
        probe_dirs = natsorted(probe_dirs, key=str)
        all_probe_dirs.append(probe_dirs)
    if not return_dates:
        return all_probe_dirs
    else:
        from dateparser.search import search_dates
        dates = [search_dates(str(folder), languages=['en']) for folder in all_probe_dirs[0]] #FIXME: currently returns none

        return all_probe_dirs, dates

def load_cluster_data(kilosort_path, use_phy=False):
    """get data from a kilosort directory. A quick and dirty version of read_phy_data()

    Parameters
    ----------
    kilosort_path : string or pathlib.Path
        the absolute path of the kilosort results directory
    use_phy : bool, optional
        load_phy labels instead of kilosort labels, by default False

    Returns
    -------
    dat
        a dictionary with keys 'spks', 'clusters', and 'labels'
    """
    data = dict()
    kilosort_path = Path(kilosort_path)
    data['spks'] = np.load(kilosort_path / 'spike_times.npy')
    data['clusters'] = np.load(kilosort_path / 'spike_clusters.npy')
    if use_phy and is_phy_curated(kilosort_path):
        cluster_filename = 'cluster_group.tsv' #this file gets modified during phy curation
    else:
        print('Using kilosort data (either there is no phy curation, or the user has specified that kilosort data should be used).')
        cluster_filename = 'cluster_KSLabel.tsv'
    data['labels'] = pd.read_csv(kilosort_path / cluster_filename, sep='\t')
    return data

def map_binary(fname,nchannels,dtype=np.int16,
               offset = 0,
               mode = 'r',nsamples = None,transpose = False):
    ''' 
    dat = map_binary(fname,nchannels,dtype=np.int16,mode = 'r',nsamples = None)
    
Memory maps a binary file to numpy array.
    Inputs: 
        fname           : path to the file
        nchannels       : number of channels
        dtype (int16)   : datatype
        mode ('r')      : mode to open file ('w' - overwrites/creates; 'a' - allows overwriting samples)
        nsamples (None) : number of samples (if None - gets nsamples from the filesize, nchannels and dtype)
    Outputs:
        data            : numpy.memmap object (nchannels x nsamples array)
See also: map_spikeglx, numpy.memmap

    Usage:
Plot a chunk of data:
    dat = map_binary(filename, nchannels = 385)
    chunk = dat[:-150,3000:6000]
    
    import pylab as plt
    offset = 40
    fig = plt.figure(figsize=(10,13)); fig.add_axes([0,0,1,1])
    plt.plot(chunk.T - np.nanmedian(chunk,axis = 1) + offset * np.arange(chunk.shape[0]), lw = 0.5 ,color = 'k');
    plt.axis('tight');plt.axis('off');

    '''
    dt = np.dtype(dtype)
    if not os.path.exists(fname):
        if not mode == 'w':
            raise(ValueError('File '+ fname +' does not exist?'))
        else:
            print('Does not exist, will create [{0}].'.format(fname))
            if not os.path.isdir(os.path.dirname(fname)):
                os.makedirs(os.path.dirname(fname))
    if nsamples is None:
        if not os.path.exists(fname):
            raise(ValueError('Need nsamples to create new file.'))
        # Get the number of samples from the file size
        nsamples = os.path.getsize(fname)/(nchannels*dt.itemsize)
    ret = np.memmap(fname,
                    mode=mode,
                    dtype=dt,
                    shape = (int(nsamples),int(nchannels)),
                    order='C', 
                    offset = offset)
    if transpose:
        ret = ret.transpose([1,0])
    return ret

def _binary_get_channel_chunk(bin_file_path, channel_idx, nchannels, offset = 0, nsamples = 30000*10, dtype = np.int16):
    itemsize = np.dtype(dtype).itemsize
    with open(bin_file_path, mode="rb") as f:
        f.seek(offset*nchannels*itemsize,os.SEEK_SET)
        chunk = np.fromfile(f,count = int(nchannels*nsamples), dtype = dtype)[channel_idx:-1:nchannels]
    return chunk

def _get_binary_multiproceessing_wrapper(chunk,path,channel_idx,nchannels):
    dat = _binary_get_channel_chunk(path,channel_idx,nchannels,chunk[0],nsamples = chunk[1]-chunk[0])
    return [chunk[0],dat]

def binary_read_single_channel(bin_file_path,channel_idx,chunksize = 30000*10):
    '''
    A function to extract the data from the binary file.
    '''
    
    from .spikeglx_utils import load_spikeglx_binary
    data, meta = load_spikeglx_binary(bin_file_path)  # TODO: this needs to be changed to work with any binary file, not just spikeglx
    chunks = chunk_indices(data, chunksize = chunksize)
    nchannels = data.shape[1]

    

    from tqdm import tqdm    
    with Pool(processes=cpu_count()) as pool: 
            res = []
            for r in tqdm(pool.imap_unordered(partial(_get_binary_multiproceessing_wrapper,
                                                      path = bin_file_path,
                                                      channel_idx = channel_idx,
                                                      nchannels = nchannels), chunks), total=len(chunks)): 
                res.append(r)
    return np.hstack([res[i][1] for i in np.argsort([r[0] for r in res])])


def concatenate_binary_files(files,output_file, fix_metadata = True):
    '''Written by Joao Couto, pnc_spks repo'''
    dat = []
    metadata = []
    files = natsorted(files)
    for f in files:
        data, meta = load_spikeglx_binary(f)
        dat.append(data)
        metadata.append(meta)
    fileSizeBytes = [m['fileSizeBytes'] for m in metadata]
    fileTimeSecs = [m['fileTimeSecs'] for m in metadata]
    # concatenate the binary file, this takes some time
    # write the files
    chunksize = 10*4096 
    pbar = tqdm(total = np.sum(fileSizeBytes))
    with open(output_file, 'wb') as outf:
        for file,size in zip(files,fileSizeBytes):
            current_pos = 0
            pbar.set_description(os.path.basename(file))
            with open(file, mode='rb') as f:
                while not current_pos == size:
                    if current_pos + chunksize < size:
                        chunk = chunksize
                    else:
                        chunk = int(size - current_pos)
                    contents = f.read(chunk)
                    outf.write(contents)
                    current_pos += chunk
                    pbar.update(chunk)
    if fix_metadata:
        _fix_metadata(output_file, files)
        
def _fix_metadata(output_file, files): # for binary concatenation 
    metadata = []
    files = natsorted(files)
    for f in files:
        _, meta = load_spikeglx_binary(f)
        metadata.append(meta)

    fileSizeBytes = [m['fileSizeBytes'] for m in metadata]
    fileTimeSecs = [m['fileTimeSecs'] for m in metadata]
        
    outmeta = Path(output_file).with_suffix('.meta')
    with open(Path(files[0]).with_suffix('.meta')) as file:
        lines = [line.rstrip() for line in file.readlines()]
    for i,line in enumerate(lines):
        if line.startswith('fileSizeBytes'):
            lines[i] = 'fileSizeBytes={0:d}'.format(int(np.sum(fileSizeBytes)))
        if line.startswith('fileTimeSecs'):
            lines[i] = 'fileTimeSecs={0:f}'.format(np.sum(fileTimeSecs))
    lines.append('concatenatedFiles='+' '.join(
        [os.path.basename(f) for f in files]))
    lines.append('concatenatedFilesOffsetBytes='+' '.join(
        [str(int(b)) for b in np.cumsum(fileSizeBytes)]))
    lines.append('concatenatedFilesOffsetTimeSecs='+' '.join(
        [str(b) for b in np.cumsum(fileTimeSecs)]))
    with open(outmeta,'w') as file:
        for line in lines:
            file.write(line + '\n')

def split_binary_file():
    '''splits binary file back into individual files'''
    raise NotImplementedError()



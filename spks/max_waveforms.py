from tqdm import tqdm
from multiprocessing import Pool,cpu_count
from functools import partial 
import numpy as np
import os
from .phy_utils import load_spikeglx_binary

class TemporaryArrayOnDisk(np.memmap):
    """TemporaryArrayOnDisk extends np.memmap in two ways
        - provides a destructor function that will delete the memmap file when all variable references to it are gone.
        - will accept an absolute binary path, or will randomly generate a filename if given a target directory

    Max Melin - 2023
    """
    def __new__(cls, fast_drive_path,*args, **kwargs):
        FILE_EXTENSION = '.bin'
        if os.path.isdir(fast_drive_path):
            import random
            import string
            FILE_STRING_LENGTH = 20 # the length of the randomly generated filename
            random_file_name = ''.join(random.choices(string.ascii_letters, k=FILE_STRING_LENGTH)) + FILE_EXTENSION
            filename = os.path.join(fast_drive_path, random_file_name)
        else:
            _, file_extension = os.path.splitext(fast_drive_path)
            assert file_extension == FILE_EXTENSION
            filename = fast_drive_path
        self = super().__new__(cls, filename, *args, **kwargs)
        return self

    def __del__(self):
        # The destructor will delete the file when the object is deleted or goes out of scope
        # but it will not handle a system exit while the kernel is running.
        if self._mmap:
            self._mmap.close()
            os.remove(self.filename)


def __par_init_waveforms(bin_file_path, waveform_mmap_path, mmap_shape, local_timestamps, local_time_indices):
    """Initialize the workers for _work_extract_waveforms"""
    global data
    global waveforms
    global chmap
    global timestamps
    global time_indices

    timestamps=local_timestamps
    time_indices=local_time_indices

    data, meta = load_spikeglx_binary(bin_file_path)
    waveforms = np.memmap(waveform_mmap_path,
                        mode='r+',
                        dtype=np.int16, 
                        order='C',
                        shape = mmap_shape)
    chmap = None
    if chmap is None: #TODO: move chmap definition to the proper place
        chmap = np.arange(data.shape[1])
    
def _work_extract_waveforms(time_indices, chunk_inds, flush_memory=True):
    """Extracts waveforms from binary file and writes them to the global variable waveforms."""
    
    spike_times = timestamps.flatten()[chunk_inds]

    holder = np.empty((len(chunk_inds),len(time_indices),len(chmap)))
    for i,s in enumerate(spike_times):
        holder[i,:,:] = np.take(data[time_indices+s,:].astype(np.int16),chmap,axis=1)
    waveforms[chunk_inds,:,:] = holder

    if flush_memory:
        waveforms.flush() #runs MUCH faster with no flush if sufficient memory, but no flush is much slower if memory is exceeded, which it usually is
    return
    
def extract_memmapped_waveforms(binfile_path, scratch_directory, nchannels, timestamps, n_spikes_per_chunk=1000, npre=30, npost=30, chmap=None):
    """Takes an array of timestamps and extracts the waveforms on all channels. Waveforms are memory mapped to a binary
    file to overcome memory limits.

    Parameters
    ----------
    binfile_path : string or Path
        absolute path to the binary file 
    scratch_directory : string or Path
        Temporary folder for saving the memory-mapped waveforms. This should be the fastest drive availible on the computer.
    nchannels : int
        the number of channels in the binary data
    timestamps : ndarray
        the timestamps (in samples) of each spike to be extracted
    n_spikes_per_chunk : int, optional
        chunk_size for parallel processing, by default 1000
    npre : int, optional
        number of samples before a spike to grab, by default 30
    npost : int, optional
        number of samples after a spike to grab, by default 30
    chmap : _type_, optional
        _description_, by default None

    Returns
    -------
    TemporaryArrayOnDisk
        An extension of np.memmap that will automatically delete the binary file when the variable goes out of scope or is deleted.
        Size: (n_timestamps, npre+npost, nchannels) and can be indexed like a numpy array.
        Deletion will not work upon a forced exit.

    Max Melin - 2023
    """
    n_chunks = timestamps.size // n_spikes_per_chunk + 1
    chunks = np.arange(n_chunks)
    time_indices = np.arange(-npre,npost,dtype=np.int16)

    chunk_inds = []
    for c in chunks:
        inds2get = np.arange(c*n_spikes_per_chunk, (c+1)*n_spikes_per_chunk)
        inds2get = inds2get[inds2get < timestamps.size] #truncate last chunk
        if len(inds2get):
            chunk_inds.append(inds2get)

    mmap_shape = (len(timestamps),npre+npost,nchannels)

    tfile = TemporaryArrayOnDisk(scratch_directory,
                                 mode='w+',
                                 dtype=np.int16, 
                                 order='C',
                                 shape = mmap_shape)
    tfile.flush()
    print(f'\nWaveforms mapped to {tfile.filename}')

    mpfunc = partial(_work_extract_waveforms, time_indices)

    ### SINGLE THREADED ###
    #__par_init_waveforms(binfile_path, temp_mmap_file, mmap_shape)
    #_ = list(map(mpfunc, tqdm(chunk_inds)))
    #global waveforms
    #del waveforms
    #######################

    ### PARALLEL PROCESSING ###
    print(f'Extracting waveforms with chunk-size {n_spikes_per_chunk}')
    with Pool(processes=cpu_count(),
              initializer=__par_init_waveforms,
              initargs=(binfile_path, tfile.filename, mmap_shape, timestamps, time_indices)) as pool:
            for _ in tqdm(pool.imap_unordered(mpfunc, chunk_inds), total=len(chunk_inds)): #for tqdm iterator
                pass
            #pool.map(mpfunc, chunk_inds)
    ############################
    return tfile

def mean_waveforms(binfile_path, scratch_directory, nchannels, all_timestamps_listed, **extract_waveforms_kwargs):
    """Take all_listed_timestamps which is a list of the timestamps for each cluster. 
    run extract_waveforms on them (or a subset) and return the mean"""
    raise NotImplementedError()
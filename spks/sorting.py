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
'''
        pass

class Kilosort25():
    def __init__():
        pass
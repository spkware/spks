from .utils import *
from .io import *



def read_spikeglx_meta(metafile):
    '''
    Read spikeGLX metadata file.
    '''
    with open(metafile,'r') as f:
        meta = {}
        for ln in f.readlines():
            tmp = ln.split('=')
            k,val = tmp
            k = k.strip()
            val = val.strip('\r\n')
            if '~' in k:
                meta[k.strip('~')] = val.strip('(').strip(')').split(')(')
            else:
                try: # is it numeric?
                    meta[k] = float(val)
                except:
                    try:
                        meta[k] = float(val) 
                    except:
                        meta[k] = val
    # Set the sample rate depending on the recording mode
    meta['sRateHz'] = meta[meta['typeThis'][:2]+'SampRate']
    try:
        parse_coords_from_spikeglx_metadata(meta)
    except Exception as err:
        print(err)
        pass
    return meta

def parse_coords_from_spikeglx_metadata(meta,shanksep = 250):
    '''
    Python version of the channelmap parser from spikeglx files.
    Adapted from the matlab from Jeniffer Colonel

    Joao Couto - 2022
    '''
    if not 'imDatPrb_type' in meta.keys():
        meta['imDatPrb_type'] = 0.0 # 3A/B probe
    probetype = int(meta['imDatPrb_type'])
    shank_sep = 250

    imro = np.stack([[int(i) for i in m.split(' ')] for m in meta['imroTbl'][1:]])
    chans = imro[:,0]
    banks = imro[:,1]
    shank = np.zeros(imro.shape[0])
    if 'snsShankMap' in meta.keys():
        connected = np.stack([[int(i) for i in m.split(':')] for m in meta['snsShankMap'][1:]])[:,3]
    else:
        connected = np.stack([[int(i) for i in m.split(':')] for m in meta['snsGeomMap'][1:]])[:,3] # recent spikeglx
    if (probetype <= 1) or (probetype == 1100) or (probetype == 1300):
        # <=1 3A/B probe
        # 1100 UHD probe with one bank
        # 1300 OPTO probe
        electrode_idx = banks*384 + chans
        if probetype == 0:
            nelec = 960;    # per shank
            vert_sep  = 20; # in um
            horz_sep  = 32;
            pos = np.zeros((nelec, 2))
            # staggered
            pos[0::4,0] = horz_sep/2       # sites 0,4,8...
            pos[1::4,0] = (3/2)*horz_sep   # sites 1,5,9...
            pos[2::4,0] = 0;               # sites 2,6,10...
            pos[3::4,0] = horz_sep         # sites 3,7,11...
            pos[:,0] = pos[:,0] + 11          # x offset on the shank
            pos[0::2,1] = np.arange(nelec/2) * vert_sep   # sites 0,2,4...
            pos[1::2,1] = pos[0::2,1]                    # sites 1,3,5...

        elif probetype == 1100:   # HD
            nelec = 384      # per shank
            vert_sep = 6    # in um
            horz_sep = 6
            pos = np.zeros((nelec,2))
            for i in range(7):
                ind = np.arange(i,nelec,8)
                pos[ind,0] = i*horz_sep
                pos[ind,1] = np.floor(ind/8)* vert_sep
        elif probetype == 1300: #OPTO
            nelec = 960;    # per shank
            vert_sep  = 20; # in um
            horz_sep  = 48;
            pos = np.zeros((nelec, 2))
            # staggered
            pos[0:-1:2,0] = 0          # odd sites
            pos[1:-1:2,0] = horz_sep   # even sites
            pos[0:-1:2,1] = np.arange(nelec/2) * vert_sep
    elif probetype == 24 or probetype == 21:
        electrode_idx = imro[:,2]
        if probetype == 24:
            banks = imro[:,2]
            shank = imro[:,1]
            electrode_idx = imro[:,4]
        nelec = 1280       # per shank; pattern repeats for the four shanks
        vert_sep  = 15     # in um
        horz_sep  = 32
        pos = np.zeros((nelec, 2))
        pos[0::2,0] = 0                              # x pos
        pos[1::2,0] = horz_sep
        pos[1::2,0] = pos[1::2,0] 
        pos[0::2,1] = np.arange(nelec/2) * vert_sep   # y pos sites 0,2,4...
        pos[1::2,1] = pos[0::2,1]                     # sites 1,3,5...
    else:
        print('ERROR [parse_coords_from_spikeglx_metadata]: probetype {0} is not implemented.'.format(probetype))
        raise NotImplementedError('Not implemented probetype {0}'.format(probetype))
    coords = np.vstack([shank*shank_sep+pos[electrode_idx,0],
                        pos[electrode_idx,1]]).T    
    idx = np.arange(len(coords))
    meta['coords'] = coords[connected==1,:]
    meta['channel_idx'] = idx[connected==1]
    return idx,coords,connected

def load_spikeglx_binary(fname, dtype=np.int16):
    ''' 
    data,meta = load_spikeglx_binary(fname,nchannels)
    
    Memory maps a spikeGLX binary file to numpy array.

    Inputs: 
        fname           : path to the file
    Outputs:
        data            : numpy.memmap object (nchannels x nsamples array)
        meta            : meta data from spikeGLX
    '''
    name = os.path.splitext(fname)[0]
    ext = '.meta'

    metafile = name + ext
    if not os.path.isfile(metafile):
        raise(ValueError('File not found: ' + metafile))
    meta = read_spikeglx_meta(metafile)
    nchans = meta['nSavedChans']
    return map_binary(fname,nchans,dtype=np.int16,mode = 'r'),meta

def load_spikeglx_mtsdecomp(fname): # loads a compressed file for on the fly decompression
    ''' 
    data,meta = load_spikeglx_mtscomp(fname)
    
    load a compressed .cbin file with mtsdecomp() for on the fly decompression.
    Use the .close() method to close the file when done

    Inputs: 
        fname           : path to the file
    Outputs:
        data            : #TODO: change
        meta            : meta data from spikeGLX
    
    Max Melin, 2023
    '''
    name, extension = os.path.splitext(fname)
    assert extension == '.cbin', 'Not a compressed file'
    ext = '.meta'

    metafile = name + ext
    if not os.path.isfile(metafile):
        raise(ValueError('File not found: ' + metafile))
    meta = read_spikeglx_meta(metafile)
    
    from mtscomp import decompress
    return decompress(Path(name).with_suffix('.ap.cbin'), Path(name).with_suffix('.ap.ch')), meta #FIXME: make compatible with all binary files (no .ap extension)

def get_npix_lfp_triggered(dat,meta,onsets,dur,tpre=1,car_subtract = True):
    srate = meta['imSampRate']
    lfp = []
    idx = np.arange(-tpre*srate,(dur + tpre)*srate,1,dtype = int)
    from tqdm import tqdm
    for i in tqdm(onsets):
        tmp = dat[int(i*srate)+idx,:]
        if car_subtract:
            tmp = (tmp.T - np.median(tmp,axis=1)).T
            tmp = tmp - np.median(tmp,axis=0)
        lfp.append(tmp)
    lfp = np.stack(lfp)
    gain = np.float32(meta['~imroTbl'][1].split(' ')[3])
    microv_per_bit = ((meta['imAiRangeMax'] - meta['imAiRangeMin'])/(2**16))/gain*1e6
    lfp *= microv_per_bit
    return lfp

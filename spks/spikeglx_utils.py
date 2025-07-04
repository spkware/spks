from .utils import *
from .io import *

def read_mux(mux_table):
    ''' 
    The channels are sampled with multiple ADC, there are multiple ADCs in the probe but not as many as channels. 

    The ADCs can't sample all channels at the same time, and sample them sequentially.
'''
    
    n_adc,n_channels_per_adc= [int(v) for v in mux_table[0].split(',')]
    # these are the channels that are sampled together
    adc_channel_groups = [[int(v) for v in group.split(' ')]
                          for group in mux_table[1:]]
    return n_adc,n_channels_per_adc,adc_channel_groups

def read_imro(imro):
    probe_type,nchannels = [int(i) for i in imro[0].split(',')][:2]
    
    if not probe_type in [0,1020,1030,1100,1120,1121,1122,1123,1200,1300, # NP 1.0-like
                          21,2003,2004, # NP 2.0, single multiplexed shank
                          24,2013,2014,  # NP 2.0, 4-shank
                          2020,2021, # Quad-probe
                          1110]:  # UHD programmable
        from warnings import warn
        warn(f'The imro table format is not supported, assuming 1.0 {probe_type}.')
    imro_table = []
    for ln in imro[1:]:
        imro_table.append([int(i) for i in ln.split(' ')])
    if probe_type in [24,2013,2014]: # NP2 4 shank
        keys = ['channel_id', 'shank_id',
                'bank_id', 'reference_id', 'electrode_id']
    elif probe_type in [21,2003,2004]: # NP2 1 shank
        keys = ['channel_id',
                'bank_id', 'reference_id', 'electrode_id']
    elif probe_type in [1110]: # UHD
        keys = ['channel_id',
                'bank_id', 'reference_id']
    else: # assume 1.0
        keys = ['channel_id',
                'bank_id',
                'reference_id',
                'ap_gain',
                'lf_gain',
                'ap_highpass']
        keys = keys[:len(imro_table[0])]
    # Add fixed gain for 2.0 probes
    if not 'ap_gain' in keys:
        keys.append('ap_gain')
        [imro_table[i].append(80) for i in range(len(imro_table))]
    if not 'shank_id' in keys:  # other probes get shank_id of 0
        keys.append('shank_id')
        [imro_table[i].append(0) for i in range(len(imro_table))]
    imro_table = pd.DataFrame(imro_table,columns=keys)
    return probe_type,nchannels,imro_table



def read_spikeglx_meta(metafile):
    '''
    Read spikeGLX metadata file.
    '''
    with open(metafile,'r') as f:
        meta = {}
        for ln in f.readlines():
            tmp = ln.split('=')
            if not len(tmp) == 2:
                continue # it would break if there is a newline in the end 
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
    if meta['typeThis'] == 'nidq':
        # skip reading imro tables and all
        return meta
    try:
        parse_coords_from_spikeglx_metadata(meta)
    except Exception as err:
        print(err)
        pass
    # convertion to uvolts
    if 'imMaxInt' in meta.keys():
        maxval = int(meta['imMaxInt'])
    else:
        maxval = 512
    if 'imAiRangeMax' in meta.keys():
        meta['conversion_factor_microV'] = 1e6*float(meta['imAiRangeMax'])/maxval
    elif 'niAiRangeMax' in meta.keys():
        meta['conversion_factor_microV'] = 1e6*float(meta['niAiRangeMax'])/32768
    if 'imroTbl' in meta.keys():
        meta['probe_type'],meta['nchannels'],meta['imro_table'] = read_imro(meta['imroTbl'])
        meta['conversion_factor_microV'] = meta['conversion_factor_microV']/meta['imro_table']['ap_gain'].values
        if 'channel_idx' in meta.keys():
            # This is so it works with the ULTRA alphas..
            if len(meta['conversion_factor_microV']) != len(meta['channel_idx']):
                meta['conversion_factor_microV'] = meta['conversion_factor_microV'][0]*np.ones_like(meta['channel_idx'])
    if 'muxTbl' in meta.keys():
        (meta['n_adc'],
         meta['n_channels_per_adc'],
         meta['adc_channel_groups']) = read_mux(meta['muxTbl'])
    #TODO deal with the NI gains
    #TODO deal with LF files.
    # deal with version 20190413
    if not 'imDatPrb_sn' in meta.keys() and 'imProbeSN' in meta.keys():
        meta['imDatPrb_sn'] = meta['imProbeSN']
    return meta

def read_geommap(tb):
    probetype,nshanks,shank_pitch,shank_width = [i for i in tb[0].split(',')]
    keys = ['shank_id','xcoord','ycoord','connected']
    table = []
    for ln in tb[1:]:
        table.append([int(i) for i in ln.split(':')])
    table = pd.DataFrame(table,columns=keys)
    table['channel_idx'] = np.arange(len(table)).astype(int)
    return table,dict(probetype = probetype,
                    n_shanks = int(nshanks),
                    shank_pitch = float(shank_pitch),
                    shank_width = float(shank_width))

def parse_coords_from_spikeglx_metadata(meta,shanksep = 250):
    '''
    Python version of the channelmap parser from spikeglx files.
    
    The 'else' is adapted from the matlab from Jeniffer Colonel

    Joao Couto - 2022
    '''

    if 'snsGeomMap' in meta.keys():
        tbl,other = read_geommap(meta['snsGeomMap'])
        coords = np.vstack(tbl[['xcoord','ycoord']].values)
        idx = tbl['channel_idx'].values
        shank = tbl['shank_id'].values
        if other['n_shanks']> 1:
            coords[:,0] = coords[:,0] + shank*other['shank_pitch']
        connected = tbl['connected'].values
    else:
        if not 'imDatPrb_type' in meta.keys():
            meta['imDatPrb_type'] = 0.0 # 3A/B probe
        probetype = int(meta['imDatPrb_type'])
        shank_sep = 250
        probe_type,nchannels,imro_table = read_imro(meta['imroTbl'])
        #imro = np.stack([[int(i) for i in m.split(' ')] for m in meta['imroTbl'][1:]])
        chans = imro_table.channel_id.values
        banks = imro_table.bank_id.values
        shank = np.zeros(len(imro_table))
        if 'snsShankMap' in meta.keys():
            connected = np.stack([[int(i) for i in m.split(':')] for m in meta['snsShankMap'][1:]])[:,3]
        else:
            connected = np.stack([[int(i) for i in m.split(':')] for m in meta['snsGeomMap'][1:]])[:,3] # recent spikeglx
        if (probetype <= 1) or (probetype == 1100) or (probetype == 1110) or (probetype == 1300):
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

            elif probetype == 1100 or probetype == 1110:   # HD
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
            electrode_idx = imro_table.electrode_id.values
            if probetype == 24:
                banks = imro_table.bank_id.values
                shank = imro_table.shank_id.values
                electrode_idx = imro_table.electrode_id.values
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
    meta['channel_shank'] = shank[connected==1]
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

import os
import ntpath
import decimal
import copy

import numpy as np
import pyfits

from util import sf, resample2d

class cslice():
  ''' 
    A datacube slice.
  '''
  def __init__(self, logger, data):
    self.logger = logger
    self.data = data

  def setRegionData(self, cslice_x_s, cslice_x_e, cslice_y_s, cslice_y_e, data, 
    append=False):
    '''
      Set the data [self.data] in the region defined by [cslice_x_s], 
      [cslice_x_e], [cslice_y_s], [cslice_y_e] to [data].
    '''
    if append:
      self.data[cslice_y_s:cslice_y_e, cslice_x_s:cslice_x_e] += \
      data[cslice_y_s:cslice_y_e, cslice_x_s:cslice_x_e]
    else:
      self.data[cslice_y_s:cslice_y_e, cslice_x_s:cslice_x_e] = \
      data[cslice_y_s:cslice_y_e, cslice_x_s:cslice_x_e]

class cube():
  '''
    A datacube.
  '''
  def __init__(self, logger):
    self.logger = logger
    self.cslices = []
  
  def append(self, s):
    '''
      Add a cslice instance to the cube.
    '''
    self.cslices.append(s)

  def write(self, args, cfg, xtra_header_keys):
    '''
      Make a hard copy of the datacube.
    '''
    header = pyfits.Header()    
    
    if os.path.exists(args.f):
      if args.c:
        os.remove(args.f)
      else:
        self.logger.debug("Output filename already exists and clobber" + \
          " is not set.")
        exit(0)
    self.logger.debug(" Writing output to " + str(args.f) + ".")
  
    data = []
    for s in self.cslices:
      data.append(s.data)
    data = np.array(data)

    # FIXME
    #header.append(('CRVAL1', -self.hfov))
    #header.append(('CDELT1', self.pscale))
    #header.append(('CRPIX1', 0.5))
    #header.append(('CUNIT1', "arcsec"))
    #header.append(('CTYPE1', "PARAM"))
    #header.append(('CRVAL2', -self.hfov))
    #header.append(('CDELT2', self.pscale))
    #header.append(('CRPIX2', 0.5))
    #header.append(('CUNIT2', "arcsec"))
    #header.append(('CTYPE2', "PARAM"))  
    #header.append(('RESAMPTO', str(cfg['RESAMPLE_TO']), 'wavelength at which images are rescaled to (m)'))
    #header.append(('SWIDTH', cfg['SLICE_WIDTH'], 'slice width (resolution units)'))
    #header.append(('HFOV', cfg['HFOV'], 'spatial HFOV (arcsec)'))
    #header.append(('PREFWAV', cfg['PUPIL_REFERENCE_WAVELENGTH'], 'wavelength at which pup. size is determined (m)'))
    #header.append(('PSAMPLE', cfg['PUPIL_SAMPLING'], 'pupil sampling grid size (px)'))
    #header.append(('PGAMMA', cfg['PUPIL_GAMMA'], 'factor by which pupil grid is padded'))
    #header.append(('RESAMPFA', cfg['RESAMPLING_FACTOR'], 'factor by which output is scaled'))
    #header.append(('SLENGTH', float(sf(p['GENERAL']['SLICE_LENGTH'], 4)), 'slice length (mm)'))
    #header.append(('ISLENGTH', float(sf(p['GENERAL']['INTER_SLICE_LENGTH'], 4)), 'inter-slice length (mm)'))
    #header.append(('SSTAGGER', float(sf(p['GENERAL']['SLICE_STAGGER'], 4)), 'slice stagger (mm)'))
    #header.append(('WAVESTA', p['GENERAL']['WAVELENGTH_START'], 'wavelength start (m)'))
    #header.append(('WAVEEND', p['GENERAL']['WAVELENGTH_END'], 'wavelength end (m)'))
    #header.append(('WAVEINT', p['GENERAL']['WAVELENGTH_INTERVAL'], 'wavelength interval (m)'))
    #header.append(('CAMPATH', ntpath.basename(p['GENERAL']['CAMERA_LENS_PATH']), 'camera lens file'))
    #header.append(('COLPATH', ntpath.basename(p['GENERAL']['COLLIMATOR_LENS_PATH']), 'collimator lens file'))
    #header.append(('CAMCONF', p['GENERAL']['CON_CAMERA'], 'camera lens configuration enumeration'))
    #header.append(('COLCONF', p['GENERAL']['CON_COLLIMATOR'], 'collimator lens configuration enumeration'))
    #header.append(('WFESAMP', p['GENERAL']['WFE_SAMPLING'], 'Zemax WFE sampling enumeration'))
    #header.append(('SEARCHD', p['GENERAL']['SEARCH_DIRECTORY'], 'file search directory'))
    #header.append(('CAMEFFL', p['GENERAL']['CAMERA_EFFL'], 'camera EFFL (mm)'))
    #header.append(('CAMWFNO', p['GENERAL']['CAMERA_WFNO'], 'camera WFNO'))
    #header.append(('EPD', p['GENERAL']['EPD'], 'entrance pupil diameter (mm)'))
    #header.append(('NSLICES', p['GENERAL']['NSLICES'], 'number of slices'))
    #header.append(('DETPITCH', str(args.d), 'detector pixel pitch (m)'))
    #header.append(('HASCOWFE', args.cow, 'has collimator WFE been added?'))
    #header.append(('HASCAWFE', args.caw, 'has camera WFE been added?'))
    #for k, v in xtra_header_keys.iteritems():
    #  header.append((k, v[0], v[1]))
    
    pyfits.writeto(args.f, data, header)


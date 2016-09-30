import os
import ntpath
import decimal
import copy

import numpy as np
import pyfits

from util import sf, resample2d

class cube():
  '''
    Datacube class.
  '''
  def __init__(self, logger, shape, resampling_im):
    self.data 			= np.zeros(shape=shape)
    self.pscale 		= resampling_im.pscale
    self.hfov			= resampling_im.getDetectorHFOV()
    self.resolution_element 	= resampling_im.resolution_element
    self.logger 		= logger
    self.cube_idx		= 0
  
  def addImage(self, image):
    if self.data.shape[1:3] != image.data.shape:
      self.cube.logger.critical(" Data cannot be added to image - shapes must be identical!")
      exit(0)
    self.data[self.cube_idx] 	= image.data
    self.cube_idx += 1
    
  def resampleAndCrop(self, resampling_factor, hfov):
    sampling_pre_rebin 		= self.pscale					# current "/px
    hfov_pre_rebin		= self.hfov
    sampling_post_rebin 	= self.resolution_element/resampling_factor	# desired "/px
    hfov_post_rebin		= hfov
    
    self.logger.debug(" Resampling data.")
    data = []
    for d in self.data:
      data.append(resample2d(d, -hfov_pre_rebin, hfov_pre_rebin, sampling_pre_rebin, -hfov_post_rebin, hfov_post_rebin, sampling_post_rebin))
    self.data = np.array(data)    
    
    self.pscale = sampling_post_rebin
    self.hfov	= hfov_post_rebin
  
  def write(self, fname, cfg, p, args, pupil_physical_diameter):
    header = pyfits.Header()    
    header.append(('CRVAL1', -self.hfov))
    header.append(('CDELT1', self.pscale))
    header.append(('CRPIX1', 0.5))
    header.append(('CUNIT1', "arcsec"))
    header.append(('CTYPE1', "PARAM"))
    header.append(('CRVAL2', -self.hfov))
    header.append(('CDELT2', self.pscale))
    header.append(('CRPIX2', 0.5))
    header.append(('CUNIT2', "arcsec"))
    header.append(('CTYPE2', "PARAM"))  
    header.append(('RESAMPTO', str(cfg['RESAMPLE_TO']), 'wavelength at which images are rescaled to (m)'))
    header.append(('SWIDTH', cfg['SLICE_WIDTH'], 'slice width (resolution units)'))
    header.append(('HFOV', cfg['HFOV'], 'spatial HFOV (arcsec)'))
    header.append(('PREFWAV', cfg['PUPIL_REFERENCE_WAVELENGTH'], 'wavelength at which pup. size is determined (m)'))
    header.append(('PSAMPLE', cfg['PUPIL_SAMPLING'], 'pupil sampling grid size (px)'))
    header.append(('PGAMMA', cfg['PUPIL_GAMMA'], 'factor by which pupil grid is padded'))
    header.append(('RESAMPFA', cfg['RESAMPLING_FACTOR'], 'factor by which output is scaled'))
    header.append(('SLENGTH', float(sf(p['GENERAL']['SLICE_LENGTH'], 4)), 'slice length (mm)'))
    header.append(('ISLENGTH', float(sf(p['GENERAL']['INTER_SLICE_LENGTH'], 4)), 'inter-slice length (mm)'))
    header.append(('SSTAGGER', float(sf(p['GENERAL']['SLICE_STAGGER'], 4)), 'slice stagger (mm)'))
    header.append(('WAVESTA', p['GENERAL']['WAVELENGTH_START'], 'wavelength start (m)'))
    header.append(('WAVEEND', p['GENERAL']['WAVELENGTH_END'], 'wavelength end (m)'))
    header.append(('WAVEINT', p['GENERAL']['WAVELENGTH_INTERVAL'], 'wavelength interval (m)'))
    header.append(('CAMPATH', ntpath.basename(p['GENERAL']['CAMERA_LENS_PATH']), 'camera lens file'))
    header.append(('COLPATH', ntpath.basename(p['GENERAL']['COLLIMATOR_LENS_PATH']), 'collimator lens file'))
    header.append(('CAMCONF', p['GENERAL']['CON_CAMERA'], 'camera lens configuration enumeration'))
    header.append(('COLCONF', p['GENERAL']['CON_COLLIMATOR'], 'collimator lens configuration enumeration'))
    header.append(('WFESAMP', p['GENERAL']['WFE_SAMPLING'], 'Zemax WFE sampling enumeration'))
    header.append(('SEARCHD', p['GENERAL']['SEARCH_DIRECTORY'], 'file search directory'))
    header.append(('CAMEFFL', p['GENERAL']['CAMERA_EFFL'], 'camera EFFL (mm)'))
    header.append(('CAMWFNO', p['GENERAL']['CAMERA_WFNO'], 'camera WFNO'))
    header.append(('EPD', p['GENERAL']['EPD'], 'entrance pupil diameter (mm)'))
    header.append(('NSLICES', p['GENERAL']['NSLICES'], 'number of slices'))
    header.append(('DETPITCH', str(args.d), 'detector pixel pitch (m)'))
    header.append(('HASCOWFE', args.cow, 'has collimator WFE been added?'))
    header.append(('HASCAWFE', args.caw, 'has camera WFE been added?'))
    header.append(('STAPUDIA', float(sf(pupil_physical_diameter, 4)), 'starting pupil physical diameter (mm)'))
    if os.path.exists(fname):
      os.remove(fname)
    self.logger.debug(" Writing output to " + str(fname) + ".")
    pyfits.writeto(fname, self.data, header)
    
class composite_image():
  '''
    Image class used to reconstruct sliced image.
  '''
  def __init__(self, logger, dim, wave, pupil):
    self.logger			= logger
    self.dim			= dim
    self.wave			= float(wave)
    self.resolution_element	= np.degrees(self.wave/(pupil.physical_entrance_diameter*pupil.physical_gsize_mfactor))*3600	# "/resolution element
    self.pscale			= self.resolution_element/pupil.gamma								# "/px
    self.detector_FOV 		= self.pscale*pupil.gsize									# deg
    self.data 			= np.zeros(dim)
    
  def addSlice(self, im):
    self.data[im.pupil.region[0]:im.pupil.region[1]] += im.getAmplitude()[im.pupil.region[0]:im.pupil.region[1]]		# if this is a slice, only adds data within slice
    
  def getExtent(self):
    return (-self.detector_FOV/2,self.detector_FOV/2,-self.detector_FOV/2,self.detector_FOV/2)
  
  def getAmplitude(self, normalise=False, scale="linear"):
    d = copy.deepcopy(np.abs(self.data))
    if scale != "linear":
      if scale == "log":
	d = np.log10(d)
      else:
	self.logger.warning(" Unrecognised scale keyword, assuming linear")
    if normalise:
      d = (d-np.min(d))/(np.max(d)-np.min(d))
    return d
import os

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
  
  def write(self, fname, verbose=False):
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
    
    if os.path.exists(fname):
      os.remove(fname)
    self.logger.debug(" Writing output to " + str(fname) + ".")
    pyfits.writeto(fname, self.data, header)
    
class composite_image():
  '''
    Image class used to reconstruct sliced image.
  '''
  def __init__(self, dim, wave, pupil):
    self.dim			= dim
    self.wave			= float(wave)
    self.resolution_element	= np.degrees(self.wave/(pupil.physical_entrance_diameter*pupil.physical_gsize_mfactor))*3600	# "/resolution element
    self.pscale			= self.resolution_element/pupil.gamma								# "/px
    self.detector_FOV 		= self.pscale*pupil.gsize									# deg
    self.data 			= np.zeros(dim)
    
  def addSlice(self, im):
    self.data[im.pupil.region[0]:im.pupil.region[1]] += im.getAmplitude()[im.pupil.region[0]:im.pupil.region[1]]		# if this is a slice, only adds data within slice
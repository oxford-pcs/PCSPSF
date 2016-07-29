import os

import numpy as np
import pyfits

from util import sf, resample2d

class cube():
  def __init__(self, logger, shape):
    self.data = np.zeros(shape=shape)
    self.logger = logger
    self.cube_idx = 0
  
  def addComposite(self, composite):
    self.data[self.cube_idx] 	= composite.data
    self.cube_idx += 1
  
  def write(self, fname, resampling_im, resampling_factor, hfov, verbose=False):
    sampling_pre_rebin 		= resampling_im.pscale					# current "/px
    hfov_pre_rebin		= resampling_im.getDetectorHFOV()
    sampling_post_rebin 	= resampling_im.resolution_element/resampling_factor	# desired "/px
    hfov_post_rebin		= hfov

    header = pyfits.Header()    
    header.append(('CRVAL1', -hfov))
    header.append(('CDELT1', sampling_post_rebin))
    header.append(('CRPIX1', 0.5))
    header.append(('CUNIT1', "arcsec"))
    header.append(('CTYPE1', "PARAM"))
    header.append(('CRVAL2', -hfov))
    header.append(('CDELT2', sampling_post_rebin))
    header.append(('CRPIX2', 0.5))
    header.append(('CUNIT2', "arcsec"))
    header.append(('CTYPE2', "PARAM"))  
    
    data = []
    for d in self.data:
      data.append(resample2d(d, -hfov_pre_rebin, hfov_pre_rebin, sampling_pre_rebin, -hfov_post_rebin, hfov_post_rebin, sampling_post_rebin))
    data = np.array(data)
    
    if os.path.exists(fname):
      os.remove(fname)
    self.logger.debug(" Writing output to " + str(fname) + ".")
    pyfits.writeto(fname, data, header)
    
  class composite():
    def __init__(self, cube, wave, pupil):
      self.cube 		= cube
      self.wave			= wave
      self.resolution_element	= np.degrees(wave/(pupil.physical_entrance_diameter*pupil.physical_gsize_mfactor))*3600		# "/resolution element
      self.pscale		= self.resolution_element/pupil.gamma								# "/px
      self.detector_FOV 	= self.pscale*pupil.gsize									# deg
      self.data 		= np.zeros(self.cube.data.shape[1:3])
    
    def add(self, im):
      if self.data.shape != im.getAmplitude().shape:
	self.cube.logger.critical(" Data cannot be added to composite - shapes must be identical.")
        exit(0)
      self.data[im.pupil.region[0]:im.pupil.region[1]] += im.getAmplitude()[im.pupil.region[0]:im.pupil.region[1]]		# if this is a slice, only adds data within slice
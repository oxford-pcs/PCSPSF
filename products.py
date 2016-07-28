import os

import numpy as np
import pyfits
from scipy.interpolate import RectBivariateSpline

from util import sf

class cube():
  def __init__(self, logger, shape):
    self.data = np.zeros(shape=shape)
    self.logger = logger
    self.params = []
    self.cube_idx = 0
  
  def addComposite(self, composite):
    self.data[self.cube_idx] 	= composite.data
    p = {}
    p['WAVE'] 			= composite.wave
    p['RESOLUTION_ELEMENT']	= composite.resolution_element
    p['PSCALE'] 		= composite.pscale
    p['DETECTOR_FOV']	 	= composite.detector_FOV
    self.params.append(p)
    self.cube_idx += 1
  
  def write(self, fname, resample, sampling_factor, fov):
    data = []
    header = pyfits.Header()
    waves = []
    for p in self.params:
      waves.append(p['WAVE'])
    sampling_post_rebin 	= self.params[np.argmin(waves)]['RESOLUTION_ELEMENT']/sampling_factor	# desired "/px
    
    header.append(('CRVAL1', -fov/2))
    header.append(('CDELT1', sampling_post_rebin))
    header.append(('CRPIX1', 0.5))
    header.append(('CUNIT1', "arcsec"))
    header.append(('CTYPE1', "PARAM"))
    header.append(('CRVAL2', -fov/2))
    header.append(('CDELT2', sampling_post_rebin))
    header.append(('CRPIX2', 0.5))
    header.append(('CUNIT2', "arcsec"))
    header.append(('CTYPE2', "PARAM"))  
    
    for d, p in zip(self.data, self.params):
      sampling_pre_rebin 	= p['PSCALE']								# current "/px
      hfov_pre_rebin		= p['DETECTOR_FOV']/2
      hfov_post_rebin		= fov/2
      
      # set up original grid
      grid_pre_rebin = np.arange(-hfov_pre_rebin, hfov_pre_rebin, sampling_pre_rebin)
      G = RectBivariateSpline(grid_pre_rebin, grid_pre_rebin, d, kx=3, ky=3)
      
      if resample=="shortest":
        self.logger.debug(" Resampling " + str(sf(p['WAVE']*10**9, 3)) + "nm to shortest wavelength (" + str(sf(np.min(waves)*10**9, 3)) + "nm)...")
        # evaluate at new grid sampling, taking into account new fov
        grid_x, grid_y = np.mgrid[-hfov_post_rebin:hfov_post_rebin:sampling_post_rebin, -hfov_post_rebin:hfov_post_rebin:sampling_post_rebin]
        H = G.ev(grid_x, grid_y)
      else:
        self.logger.debug(" Not resampling data (are you sure you want this?)")
        # evaluate at old grid sampling, taking into account new fov
        grid_x, grid_y = np.mgrid[-hfov_post_rebin:hfov_post_rebin:sampling_pre_rebin, -hfov_post_rebin:hfov_post_rebin:sampling_pre_rebin]
        H = G.ev(grid_x, grid_y)	
      data.append(H)
    data = np.array(data)
    
    if os.path.exists(fname):
      os.remove(fname)
    self.logger.debug(" Writing output... ")
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
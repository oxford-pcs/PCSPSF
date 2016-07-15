#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from util import sf

class pupil(object):
  def __init__(self, logger, sampling, gamma, verbose=True):
    self.logger		 	= logger
    self.sampling 		= sampling					# sampling size for pupil (pixels)
    self.gamma 			= gamma						# factor by which to pad the grid (==number of pixels per resolution element in image plane)
    self.gsize			= self.sampling*self.gamma			# size of padded grid in pixels
    self.physical_gsize		= None						# physical size of padded grid in units of [physical_gsize_unit]
    self.pupil_plate_scale	= None						# [physical_gsize_unit/px]
    self.physical_gsize_unit	= None						# physical unit of grid
    self.verbose		= verbose
    self.data			= None
    
  def _setup(self):
    mag = np.ones((self.gsize,self.gsize))
    phase = np.zeros(mag.shape)
    
    re = mag * np.cos(phase)
    im = mag * np.sin(phase)
    
    self.data = re + 1j * im
  
  def addToPhase(self, p):
    mag = self.getAmplitude()
    phase = self.getPhase() + p
    
    re = mag * np.cos(phase)
    im = mag * np.sin(phase)

    self.data = re + 1j * im

  def getData(self, npix=None):
    return self.data
  
  def getAmplitude(self):
    return np.abs(self.data)
    
  def getPhase(self):
    return np.angle(self.data)
   
  def getExtent(self):
    ''' 
      Returns an extent with real units for image plots.
    '''
    return (-self.physical_gsize/2, self.physical_gsize/2, -self.physical_gsize/2, self.physical_gsize/2)
  
  def toConjugateImage():
    '''
      This function constructs an image of the pupil.
    '''
    pass
  
class circular(pupil): 
  def __init__(self, logger, sampling, gamma, rad, rad_unit, verbose=True):
    super(circular, self).__init__(logger, sampling, gamma, verbose)
    self.rad 					= rad
    self.physical_gsize_unit 			= rad_unit
    self.physical_entrance_diameter		= self.rad*2					# physical diameter of entrance pupil in units of [physical_gsize_unit]
    self.physical_gsize 			= self.physical_entrance_diameter*self.gamma	
    self.pupil_plate_scale			= self.physical_entrance_diameter/sampling
    
    # translate unit to numerical quantity so output scale is physically meaningful
    if self.physical_gsize_unit == "m":
      self.physical_gsize_mfactor = 1
    elif self.physical_gsize_unit == "mm":
      self.physical_gsize_mfactor = 1e-3
    else:
      self.logger.warning(" Unrecognised pupil radius unit, assuming mm.")
      self.physical_gsize_mfactor = 1e-3

    if self.verbose:
      self.logger.debug(" Entrance pupil diameter is " + str(self.physical_entrance_diameter) + self.physical_gsize_unit)
      self.logger.debug(" With pupil sampling of " + str(self.sampling) + "x" + str(self.sampling) + ", this corresponds to a pupil plate scale of " + 
			sf(self.pupil_plate_scale, 2) + self.physical_gsize_unit + "/px.")
      
    # initialise data
    self._setup() 
   
  def _setup(self):
    y, x = np.ogrid[-self.gsize/2:self.gsize/2, -self.gsize/2:self.gsize/2]
    mask = x*x + y*y <= (self.sampling/2)*(self.sampling/2)
    mag = np.zeros((self.gsize, self.gsize))
    mag[mask] = 1
    
    phase = np.zeros(mag.shape)
    
    re = mag * np.cos(phase)
    im = mag * np.sin(phase)
    
    self.data = re + 1j * im
    
  def toConjugateImage(self, wave, shift=True):   
    i_data = np.fft.fft2(self.data)
    if shift:
      i_data = np.fft.fftshift(i_data)   
    return self.conjugateImage(self, i_data, wave)

  class conjugateImage():
    def __init__(self, pupil, i_data, wave):
      self.pupil = pupil
      self.wave	= wave
      self.data = i_data
      
      self.im_resolution_element	= np.degrees(self.wave/(self.pupil.physical_entrance_diameter*self.pupil.physical_gsize_mfactor))*3600	# "/resolution element
      self.im_pscale			= self.im_resolution_element/self.pupil.gamma								# "/px
      self.detector_FOV 		= self.im_pscale*self.pupil.gsize									# deg
      self.airy_disk_d			= 2.44*self.im_resolution_element									# "
      if self.pupil.verbose:
	self.pupil.logger.debug(" At a wavelength of " + str(self.wave*10**9) + "nm, a circular aperture would have the following properties:")
	self.pupil.logger.debug(" -> " + sf(self.im_resolution_element, 2) + "\"" + " per resolution element (λ/D)")
	self.pupil.logger.debug(" -> " + sf(self.im_pscale, 2) + "\" per pixel (with γ=" + str(self.pupil.gamma) + " pixels per resolution element)")
	self.pupil.logger.debug(" -> a detector FoV of " + sf(self.detector_FOV, 2) + "\"")
	self.pupil.logger.debug(" -> an airy disk diameter of " + sf(self.airy_disk_d, 2) + "\"")
     
    def getDetectorHFOV(self):
      return self.detector_FOV/2.

    def getAmplitude(self):
      return np.abs(self.data)
    
    def getPhase(self):
      return np.angle(self.data)
    
    def getAmplitudeScaledByAiryDiameters(self, n_airy_diameters):
      '''
        Constructs a scaled image with only [n_airy_diameters] shown. Useful for plotting.
        
        Returns data and scaled HFOV of the detector.
      '''
      im_npix 			= int(np.ceil(self.pupil.gamma*2.44*n_airy_diameters))	# limit the number of pixels in the image
      extent_scale_factor	= float(self.pupil.gsize)/im_npix			# corresponding scale factor to scale extent
      detector_HFOV_scaled 	= (self.getDetectorHFOV())/extent_scale_factor		# HFOV of detector, scaled down by [scale]   
      if self.pupil.verbose:
	self.pupil.logger.debug(" Using innermost " + str(im_npix) + " pixels, giving a detector scaled HFOV of " + str(detector_HFOV_scaled) + "\".")
	
      return self.getAmplitude()[(self.pupil.gsize/2)-(im_npix/2):(self.pupil.gsize/2)+(im_npix/2), (self.pupil.gsize/2)-(im_npix/2):(self.pupil.gsize/2)+(im_npix/2)], detector_HFOV_scaled
    
    def takeSlice(self, width_el):
      '''
	Takes a slice of width [width_el] resolution elements from centre.
	
	Pads the rest of the array with zeros.
      '''
      self.half_slice_width = int(self.pupil.gamma*width_el)		# in pixels
      if self.half_slice_width % 2 != 0:
	self.pupil.logger.critical(" Half slice width is not an even number of pixels!")
	exit(0)
      if self.pupil.verbose:
	self.pupil.logger.debug(" Taking slice of width " + str(width_el) + "γ = " + str(self.half_slice_width) + " pixels.")  
      self.data[0:(self.pupil.gsize/2)-(self.half_slice_width/2)] = 0
      self.data[(self.pupil.gsize/2)+(self.half_slice_width/2):] = 0   
      
    def toConjugatePupil(self, shift=False):   
      self.pupil.data = np.fft.ifft2(self.data)
      if shift:
	self.pupil.data = np.fft.fftshift(self.pupil.data)   
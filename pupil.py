#!/usr/local/bin/python
# -*- coding: utf-8 -*-
from copy import deepcopy

import numpy as np

from util import sf

class pupil(object):
  def __init__(self, logger, camera, sampling, gamma, verbose):
    self.logger		 	= logger
    self.camera			= camera
    self.sampling 		= sampling					# sampling size for pupil (pixels)
    self.gamma 			= gamma						# factor by which to pad the grid (==number of pixels per resolution element in image plane)
    self.gsize			= self.sampling*self.gamma			# size of padded grid in pixels
    self.physical_gsize		= None						# physical size of padded grid in units of [physical_gsize_unit]
    self.pupil_plate_scale	= None						# [physical_gsize_unit/px]
    self.physical_gsize_unit	= None						# physical unit of grid
    self.data			= None
    
  def _setup(self):
    mag = np.ones((self.gsize,self.gsize))
    phase = np.zeros(mag.shape)
    
    re = mag * np.cos(phase)
    im = mag * np.sin(phase)
    
    self.data = re + 1j * im
  
  def addToPhase(self, p):
    mag = self.getAmplitude()
    phase = self.getPhase()+p

    re = mag * np.cos(phase)
    im = mag * np.sin(phase)

    self.data = re + 1j * im

  def getData(self, npix=None):
    return self.data

  def getAmplitude(self, power=False, shift=False, scale="linear", normalise=False):
    d = deepcopy(np.abs(self.data))
    if power:
      d = d**2
    if scale != 'linear':
      if scale == 'log':
	d = np.log10(d)
      else:
	self.logger.warning(" Unrecognised scale keyword, assuming linear")
    if normalise:
      d = (d-np.min(d))/(np.max(d)-np.min(d))
    if shift:
      d = np.fft.fftshift(d)
    return d
  
  def getPhase(self, shift=False):
    if shift:
      return np.angle(np.fft.fftshift(self.data))
    return np.angle(self.data)
  
  def getRealComponent(self, shift=False, normalise=False):
    d = deepcopy(self.data)
    if shift:
      d = np.fft.fftshift(d)
    re = np.real(d)
    if normalise:
      re = (re-np.min(re))/(np.max(re)-np.min(re))
    return re
    
  def getImagComponent(self, shift=False, normalise=False):
    d = deepcopy(self.data)
    if shift:
      d = np.fft.fftshift(d)
    im = np.imag(d)
    if normalise:
      im = (im-np.min(im))/(np.max(im)-np.min(im))
    return im
      
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
  def __init__(self, logger, camera, sampling, gamma, rad, physical_gsize_unit, verbose=False):
    super(circular, self).__init__(logger, camera, sampling, gamma, verbose)
    self.rad 					= rad
    self.physical_gsize_unit 			= physical_gsize_unit
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

    if verbose:
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
    self.data = np.fft.fftshift(self.data)   
    
  def toConjugateImage(self, wave, shift=True, verbose=False):   
    i_data = np.fft.fft2(self.data)
    if shift:
      i_data = np.fft.fftshift(i_data)   
    return self.conjugateImage(self, i_data, wave, verbose)

  class conjugateImage():
    def __init__(self, pupil, i_data, wave, verbose):
      self.pupil = pupil
      self.wave	= wave
      self.data = i_data
      
      self.resolution_element	= np.degrees(self.wave/(self.pupil.physical_entrance_diameter*self.pupil.physical_gsize_mfactor))*3600	# "/resolution element
      self.pscale		= self.resolution_element/self.pupil.gamma								# "/px
      self.detector_FOV 	= self.pscale*self.pupil.gsize										# deg
      self.airy_disk_d		= 2.44*self.resolution_element										# "
      
      self.pupil.camera.populateSpatialParameters(self)
      
      if verbose:
	self.pupil.logger.debug(" At a wavelength of " + str(self.wave*10**9) + "nm, a system with a focal ratio of " + str(self.pupil.camera.wfno) + " with a circular aperture would have the following properties:")
	self.pupil.logger.debug(" -> " + sf(self.resolution_element, 4) + "\"" + " (" + sf(self.pupil.camera.s_resolution_element, 4) + "μm)" + " per resolution element λ/D")
	self.pupil.logger.debug(" -> " + sf(self.pscale, 4) + "\"" + " (" + sf(self.pupil.camera.s_pscale, 4) + "μm)" + " per pixel with γ=" + str(self.pupil.gamma) + " pixels per resolution element")
	self.pupil.logger.debug(" -> a detector FoV of " + sf(self.detector_FOV, 4) + "\"" + " (" + sf(self.pupil.camera.s_detector_FOV, 6) + "μm)")
	self.pupil.logger.debug(" -> an airy disk diameter of " + sf(self.airy_disk_d, 4) + "\"" + " (" + sf(self.pupil.camera.s_airy_disk_d, 4) + "μm)")
     
      self.slice_number = None
     
    def getDetectorHFOV(self):
      return self.detector_FOV/2.

    def getAmplitude(self, power=False, shift=False, normalise=False, scale="linear"):
      d = deepcopy(np.abs(self.data))
      if power:
        d = d**2
      if scale != "linear":
	if scale == "log":
	  d = np.log10(d)
	else:
	  self.pupil.logger.warning(" Unrecognised scale keyword, assuming linear")
      if normalise:
	d = (d-np.min(d))/(np.max(d)-np.min(d))
      if shift:
	d = np.fft.fftshift(d)
      return d
    
    def getPhase(self, shift=False):
      if shift:
        return np.angle(np.fft.fftshift(self.data))
      return np.angle(self.data)
    
    def getRealComponent(self, shift=False, normalise=False):
      d = deepcopy(self.data)
      if shift:
	d = np.fft.fftshift(d)
      re = np.real(d)
      if normalise:
	re = (re-np.min(re))/(np.max(re)-np.min(re))
      return re
      
    def getImagComponent(self, shift=False, normalise=False):
      d = deepcopy(self.data)
      if shift:
	d = np.fft.fftshift(d)
      im = np.imag(d)
      if normalise:
	im = (im-np.min(im))/(np.max(im)-np.min(im))
      return im
      
    def getAmplitudeScaledByAiryDiameters(self, n_airy_diameters, power=False, shift=False, normalise=False, scale="linear", verbose=False):
      '''
        Constructs a scaled image with only [n_airy_diameters] shown. Useful for plotting.
        
        Returns data and scaled HFOV of the detector.
      '''
      im_npix 			= int(np.ceil(self.pupil.gamma*2.44*n_airy_diameters))	# limit the number of pixels in the image
      extent_scale_factor	= float(self.pupil.gsize)/im_npix			# corresponding scale factor to scale extent
      detector_HFOV_scaled 	= (self.getDetectorHFOV())/extent_scale_factor		# HFOV of detector, scaled down by [scale]   
      if verbose:
	self.pupil.logger.debug(" Using innermost " + str(im_npix) + " pixels, giving a detector scaled HFOV of " + str(detector_HFOV_scaled) + "\".")

      return self.getAmplitude(power, shift, normalise, scale)[(self.pupil.gsize/2)-(im_npix/2):(self.pupil.gsize/2)+(im_npix/2), (self.pupil.gsize/2)-(im_npix/2):(self.pupil.gsize/2)+(im_npix/2)], detector_HFOV_scaled
    
    def takeSlice(self, width_el, offset=0, slice_number=1, verbose=False):
      '''
	Takes a slice of width [width_el] resolution element, offset by [offset] resolution elements from the centre.
	
	Pads the rest of the array with zeros.
      '''
      self.half_slice_width = int(self.pupil.gamma*width_el)		# in pixels
      self.offset_res = int(self.pupil.gamma*offset)			# in pixels
      try:
	assert self.half_slice_width % 2 == 0
	assert self.offset_res % 2 == 0
      except AssertionError:
	self.pupil.logger.critical(" Either the half slice width or offset does not correspond to an even number of pixels!")
	exit(0)
      if verbose:
	self.pupil.logger.debug(" Taking slice of width " + str(width_el) + "γ = " + str(self.half_slice_width) + " pixels with an offset of " + str(self.offset_res) + " pixels.")  
      self.data[0:(self.pupil.gsize/2)-(self.half_slice_width/2)+self.offset_res] = 0
      self.data[(self.pupil.gsize/2)+(self.half_slice_width/2)+self.offset_res:] = 0   
      
      self.slice_number = slice_number

    def toConjugatePupil(self, ishift=True, verbose=False):   
      new_pupil = circular(self.pupil.logger, self.pupil.camera, self.pupil.sampling, self.pupil.gamma, self.pupil.rad, self.pupil.physical_gsize_unit, verbose)
      new_pupil.data = deepcopy(self.data)
      if ishift:
	new_pupil.data = np.fft.ifftshift(new_pupil.data)   
      new_pupil.data = np.fft.ifft2(new_pupil.data)
      return new_pupil
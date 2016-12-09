#!/usr/local/bin/python
# -*- coding: utf-8 -*-
from copy import deepcopy

import numpy as np
import pylab as plt

from util import sf, resample2d

class pupil(object):
  '''
    This class provides abstraction for all the fields and methods 
    that are non-specific to the geometry of a given pupil.
  '''
  def __init__(self, logger, camera, sampling, gamma, verbose, data=None):
    self.logger		 	= logger
    self.camera			= camera
    self.sampling 		= sampling					# sampling size for pupil (pixels)
    self.gamma 			= gamma						# factor by which to pad the grid (=px/resolution element)
    self.physical_gsize		= None						# physical size of padded grid in units of mm
    self.pupil_plate_scale	= None						# [mm/px]
    
    self.image_slicing_region	= None
    
    if data is None:								# if there's no data..
      self.gsize	= self.sampling*self.gamma				# .. set size of padded grid in pixels
      self._setup()								# .. set up a new data grid
    else:
      self.gsize	= data.shape[0]
      self.data		= data
    
  def _setup(self):
    '''
      Instantiate [self.data] with zero phase and unity magnitude
      across the array.
    '''
    mag = np.ones((self.gsize,self.gsize))
    phase = np.zeros(mag.shape)
    
    re = mag * np.cos(phase)
    im = mag * np.sin(phase)
    
    self.data = re + 1j * im
  
  def addToPhase(self, p):
    '''
      Add a phase difference to the phase of [self.data], and reform
      as a complex number.
    '''
    mag = self.getAmplitude()
    phase = self.getPhase()+p

    re = mag * np.cos(phase)
    im = mag * np.sin(phase)

    self.data = re + 1j * im
    
  def convolve(self, p, frac_of_max_intensity=1E-3):
    '''
      Convolution in fourier space, equivalent to addToPhase() but also 
      takes into account the magnitude of [self.data] by setting the magnitude 
      of a continuous positive-valued region along the x axis to 1. The 
      region start/end is defined by finding the coordinate where a fraction 
      of the peak intensity drops below [frac_of_max_intensity].
      
      The logic here is sensitive to both the nature of the pupil, and whether 
      the components of the image are dc-centered. The following assumes a 
      non-dc centered diffracted slice that forms an IMAGE like:
      
      ^ y
      |xx   xx
      |xx   xx
      |xx   xx
      |xx   xx
      |xx___xx 
               >x
               
      i.e. where diffraction has occurred across (y) the slice. 
    '''
    mag = self.getAmplitude()					# get the current amplitude
       
    mag_median_profile_x = np.median(mag, axis=0)		# take the median along x
    pr = np.max(mag_median_profile_x)				# find the maximum value of this median profile
    lim = pr*frac_of_max_intensity				# calculate the threshold
    
    lo = np.min(np.where(mag_median_profile_x<lim))		# lowest x
    hi = np.max(np.where(mag_median_profile_x<lim))		# highest x
    
    mag_w = np.zeros(shape=mag.shape)				
    mag_w[:, 0:lo] = 1
    mag_w[:, hi:] = 1
    
    re = mag_w * np.cos(p)
    im = mag_w * np.sin(p)
    
    wfe = re + 1j * im   

    self.data = self.data*wfe

  def getAmplitude(self, power=False, shift=False, scale="linear", normalise=False):
    '''
      Get the amplitude of [self.data], or convert to a variety of formats.
      
      Returns a data array of the same shape as [self.data].
    '''
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
    '''
      Get the phase of [self.data].
      
      Returns a data array of the same shape as [self.data].
    '''
    if shift:
      return np.angle(np.fft.fftshift(self.data))
    return np.angle(self.data)
  
  def getRealComponent(self, shift=False, normalise=False):
    '''
      Returns the real component of [self.data].
    '''
    d = deepcopy(self.data)
    if shift:
      d = np.fft.fftshift(d)
    re = np.real(d)
    if normalise:
      re = (re-np.min(re))/(np.max(re)-np.min(re))
    return re
    
  def getImagComponent(self, shift=False, normalise=False):
    '''
      Returns the imaginary component of [self.data].
    '''
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
  
  def toConjugateImage(self, wave, shift=True, verbose=False):
    '''
      This function constructs an image of the pupil.
    '''
    i_data = np.fft.fft2(self.data)
    if shift:
      i_data = np.fft.fftshift(i_data)   
    return self.conjugateImage(self, i_data, wave, verbose)
  
  class conjugateImage():
    '''
      Dummy class that will be overriden by inherited classes.
    '''
    pass
  
class circular(pupil): 
  '''
    Circular pupil class.
    
    The sliced version of this class is circular_slice.
  '''
  def __init__(self, logger, camera, sampling, gamma, rad, verbose=False, data=None):
    super(circular, self).__init__(logger, camera, sampling, gamma, verbose, data)
    self.rad 					= rad
    self.physical_entrance_diameter		= self.rad*2					# physical diameter of entrance pupil in units of [mm]
    self.physical_gsize 			= self.physical_entrance_diameter*self.gamma	
    self.pupil_plate_scale			= self.physical_gsize/self.gsize

    # translate unit to numerical quantity so output scales are physically meaningful
    self.physical_gsize_mfactor = 1e-3

    if verbose:
      self.logger.debug(" Entrance pupil diameter is " + str(sf(self.physical_entrance_diameter, 3)) + 'mm')
      self.logger.debug(" With pupil sampling of " + str(sf(self.sampling, 4)) + "x" + str(sf(self.sampling, 4)) + ", this corresponds to a pupil plate scale of " + 
			sf(self.pupil_plate_scale, 2) + "mm/px")
   
    # this is just to maintain some conformity with child slicing class circular_slice
    self.region					= (0, self.data.shape[0])
    self.number					= -1
   
  def _setup(self):
    '''
      Instantiate [self.data] with zero phase and unity magnitude for a circle 
      of radius [rad] across the array.
    '''
    y, x = np.ogrid[-self.gsize/2:self.gsize/2, -self.gsize/2:self.gsize/2]
    mask = x*x + y*y <= (self.sampling/2)*(self.sampling/2)
    mag = np.zeros((self.gsize, self.gsize))
    mag[mask] = 1
    
    phase = np.zeros(mag.shape)
    
    re = mag * np.cos(phase)
    im = mag * np.sin(phase)
    
    self.data = re + 1j * im
    self.data = np.fft.fftshift(self.data)   
    
  class conjugateImage(object):
    '''
      The corresponding image class for a circular pupil.
    '''    
    def __init__(self, pupil, i_data, wave, verbose):
      self.pupil = pupil
      self.wave	= float(wave)
      self.data = i_data
      self.is_sliced = False 
      
      self.resolution_element	= np.degrees(self.wave/(self.pupil.physical_entrance_diameter*self.pupil.physical_gsize_mfactor))*3600		# "/resolution element
      self.pscale		= self.resolution_element/self.pupil.gamma									# "/px
      self.detector_FOV 	= self.pscale*self.pupil.gsize											# deg
      self.airy_disk_d		= 2.44*self.resolution_element											# "
      
      self.pupil.camera.populateSpatialParameters(self)
      
      if verbose:
	self.pupil.logger.debug(" At a wavelength of " + sf(self.wave*10**9, 4) + "nm, a system with a focal ratio of " + sf(self.pupil.camera.wfno, 3) + " with a circular aperture would have the following properties:")
	self.pupil.logger.debug(" -> " + sf(self.resolution_element, 4) + "\"" + " (" + sf(self.pupil.camera.s_resolution_element, 4) + "μm)" + " per resolution element λ/D")
	self.pupil.logger.debug(" -> " + sf(self.pscale, 4) + "\"" + " (" + sf(self.pupil.camera.s_pscale, 4) + "μm)" + " per pixel, with γ=" + str(sf(self.pupil.gamma, 2)) + " pixels per resolution element")
	self.pupil.logger.debug(" -> a detector FoV of " + sf(self.detector_FOV, 4) + "\"" + " (" + sf(self.pupil.camera.s_detector_FOV, 6) + "μm)")
	self.pupil.logger.debug(" -> an airy disk diameter of " + sf(self.airy_disk_d, 4) + "\"" + " (" + sf(self.pupil.camera.s_airy_disk_d, 4) + "μm)")
     
      self.slice_number = None
       
    def getDetectorHFOV(self):
      '''
        Returns the detector half field of view.
      '''
      return self.detector_FOV/2.

    def getAmplitude(self, power=False, shift=False, normalise=False, scale="linear"):
      '''
        Get the amplitude of [self.data], or convert to a variety of formats.
      
        Returns a data array of the same shape as [self.data].
     '''
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
      '''
       Get the phase of [self.data].
      
        Returns a data array of the same shape as [self.data].
      '''
      if shift:
        return np.angle(np.fft.fftshift(self.data))
      return np.angle(self.data)
    
    def getRealComponent(self, shift=False, normalise=False):
      '''
        Returns the real component of [self.data].
      '''
      d = deepcopy(self.data)
      if shift:
	d = np.fft.fftshift(d)
      re = np.real(d)
      if normalise:
	re = (re-np.min(re))/(np.max(re)-np.min(re))
      return re
      
    def getImagComponent(self, shift=False, normalise=False):
      '''
        Returns the imaginary component of [self.data].
      '''
      d = deepcopy(self.data)
      if shift:
	d = np.fft.fftshift(d)
      im = np.imag(d)
      if normalise:
	im = (im-np.min(im))/(np.max(im)-np.min(im))
      return im
      
    def getAmplitudeScaledByAiryDiameters(self, n_airy_diameters, power=False, shift=False, normalise=False, scale="linear", verbose=False):
      '''
        Returns a scaled image with only [n_airy_diameters] shown. Useful for plotting.
        
        Returns data and scaled HFOV of the detector.
      '''
      im_npix 			= int(np.ceil(self.pupil.gamma*2.44*n_airy_diameters))	# limit the number of pixels in the image
      extent_scale_factor	= float(self.pupil.gsize)/im_npix			# corresponding scale factor to scale extent
      detector_HFOV_scaled 	= (self.getDetectorHFOV())/extent_scale_factor		# HFOV of detector, scaled down by [scale]   
      if verbose:
	self.pupil.logger.debug(" Using innermost " + str(im_npix) + " pixels, giving a detector scaled HFOV of " + str(detector_HFOV_scaled) + "\".")

      return self.getAmplitude(power, shift, normalise, scale)[(self.pupil.gsize/2)-(im_npix/2):(self.pupil.gsize/2)+(im_npix/2), (self.pupil.gsize/2)-(im_npix/2):(self.pupil.gsize/2)+(im_npix/2)], detector_HFOV_scaled
    
    def sliceUp(self, width_el, gamma, offset=0, slice_number=0, verbose=False):
      '''
	Takes a slice of width [width_el] resolution elements (taken at the resample_to gamma to ensure a uniform
	slice width), offset by [offset] resolution elements from the centre, and pads the rest of the array with zeros.
	
	This will permanently change the data and set the is_sliced flag.
      '''
      half_slice_width 	= int(gamma*width_el)		# in pixels
      offset_res 	= int(gamma*offset)		# in pixels
      
      self.slice_number = slice_number
      self.slice_region	= ((self.pupil.gsize/2)-(half_slice_width/2)+offset_res, (self.pupil.gsize/2)+(half_slice_width/2)+offset_res)
      try:
	assert half_slice_width % 2 == 0
	assert offset_res % 2 == 0
      except AssertionError:
	self.pupil.logger.critical(" Either the half slice width or offset does not correspond to an even number of pixels!")
	exit(0)
      if verbose:
	self.pupil.logger.debug(" Taking slice of width " + str(width_el) + "γ = " + str(half_slice_width) + " pixels with an offset of " + str(offset_res) + " pixels.")  
      data = deepcopy(self.data)
      data[0:self.slice_region[0]] = 0
      data[self.slice_region[1]:] = 0   
      self.data = data
      self.is_sliced = True

    def toConjugatePupil(self, ishift=True, verbose=False):  
      '''
        Returns a new instance of either circular or circular_pupil, depending on whether
        the is_sliced flag has been set.
      '''
      if ishift:
	self.data = np.fft.ifftshift(self.data)   
      self.data = np.fft.ifft2(self.data)
      if self.is_sliced:
        new_pupil = circular_slice(self.pupil.logger, self.pupil.camera, self.pupil.sampling, self.pupil.gamma, 
				    self.pupil.rad, self.slice_number, self.slice_region, verbose, self.data)
      else:
	new_pupil = circular(self.pupil.logger, self.pupil.camera, self.pupil.sampling, self.pupil.gamma, self.pupil.rad, verbose, self.data)

      return new_pupil
    
    def resample(self, new_pscale, new_hfov, verbose=False):
      '''
          Resample data to new pixel scale and fov.
      '''
      if np.isclose(self.pscale, new_pscale) and np.isclose(self.getDetectorHFOV(), new_hfov):
	self.pupil.logger.debug(" Resampling not required for this wavelength.")
      else:
	self.pupil.logger.debug(" Data needs to be resampled from " + sf(self.pscale, 4) + " to " + sf(new_pscale, 4) + "\" per pixel.")
	if np.issubdtype(self.data.dtype, np.complexfloating):
	  self.pupil.logger.debug(" Resampling complex data requires discarding the imaginary part. This only makes sense if the phase is zero.")
	  if np.allclose(np.imag(self.data), 0):
	    self.pupil.logger.debug(" Checking to see if all imaginary parts are zero... OK")
	  else:
	    self.pupil.logger.critical(" Checking to see if all imaginary parts are zero... FAIL")
	    exit(0)
	  self.data = np.real(self.data)
	
	# resample
	self.data = resample2d(self.data, -self.getDetectorHFOV(), self.getDetectorHFOV(), self.pscale, -new_hfov, new_hfov, new_pscale)
        
        # convert back to complex
        self.data = self.data.astype(dtype=complex)
        
        # change some pupil parameters now we've rescaled
        self.pupil.gamma 		= self.resolution_element/new_pscale				# note this will now no longer be an integer
        self.pupil.physical_gsize 	= self.pupil.physical_entrance_diameter*self.pupil.gamma
        self.pupil.sampling		= self.pupil.gsize/self.pupil.gamma
	self.pupil.pupil_plate_scale	= self.pupil.physical_gsize/self.pupil.gsize
        
        # change some image parameters now we've rescaled
        self.pscale 		= new_pscale
        self.detector_FOV 	= self.pscale*self.pupil.gsize	
        
        # recalculate camera properties
        self.pupil.camera.populateSpatialParameters(self)
        
        if verbose:
	  self.pupil.logger.debug(" Resampled data now has the following properties: ")
	  self.pupil.logger.debug(" -> " + sf(self.resolution_element, 4) + "\"" + " (" + sf(self.pupil.camera.s_resolution_element, 4) + "μm)" + " per resolution element λ/D")
	  self.pupil.logger.debug(" -> " + sf(self.pscale, 4) + "\"" + " (" + sf(self.pupil.camera.s_pscale, 4) + "μm)" + " per pixel, with γ=" + str(sf(self.pupil.gamma, 2)) + " pixels per resolution element")
	  self.pupil.logger.debug(" -> a detector FoV of " + sf(self.detector_FOV, 4) + "\"" + " (" + sf(self.pupil.camera.s_detector_FOV, 6) + "μm)")
	  self.pupil.logger.debug(" -> an airy disk diameter of " + sf(self.airy_disk_d, 4) + "\"" + " (" + sf(self.pupil.camera.s_airy_disk_d, 4) + "μm)")
	  
class circular_slice(circular):
  '''
    This child class of circular has two extra fields on init, with details
    pertaining to the slice taken.
  '''
  def __init__(self, logger, camera, sampling, gamma, rad, number, region, verbose, data):
    super(circular_slice, self).__init__(logger, camera, sampling, gamma, rad, verbose, data)
    self.number		= number
    self.region 	= region
    
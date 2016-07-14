#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from util import sf

class circle():
  def __init__(self, logger, sampling, gamma, rad, rad_unit, wave, verbose=True):
    self.logger					= logger
    self.sampling 				= sampling
    self.gamma 					= gamma
    self.rad 					= rad
    self.rad_unit 				= rad_unit
    self.wave					= wave
    self.gsize					= self.sampling*self.gamma			# size of (oversized) grid in pixels
    self.physical_entrance_diameter		= self.rad*2					# physical diameter of entrance pupil in units of [rad_unit]
    self.physical_gsize 			= self.physical_entrance_diameter*self.gamma	# physical size of oversized pupil in units of [rad_unit]
    self.pupil_plate_scale			= self.physical_entrance_diameter/sampling
    self.space					= "pupil"
    self.verbose				= verbose

    # identify unit so output scale is physically meaningful
    if rad_unit == "m":
      self.mfactor	= 1
    elif rad_unit == "mm":
      self.mfactor	= 1e-3
    else:
      self.logger.warning(" Unrecognised pupil radius unit, assuming metres.")
      self.mfactor	= 1
          
    self.im_resolution_element			= np.degrees(self.wave/(self.physical_entrance_diameter*self.mfactor))*3600	# "/resolution element
    self.im_pscale				= self.im_resolution_element/self.gamma						# "/px
    self.detector_FOV 				= self.im_pscale*self.gsize							# deg
    self.airy_disk_d				= 2.44*self.im_resolution_element

    if self.verbose:
      self.logger.debug(" Entrance pupil diameter is " + str(self.physical_entrance_diameter) + self.rad_unit)
      self.logger.debug(" With pupil sampling of " + str(self.sampling) + "x" + str(self.sampling) + ", this corresponds to a pupil plate scale of " + 
			sf(self.pupil_plate_scale, 2) + self.rad_unit + "/px.")
      self.logger.debug(" At a wavelength of " + str(self.wave*10**9) + "nm, this corresponds to:")
      self.logger.debug(" -> " + sf(self.im_resolution_element, 2) + "\"" + " per resolution element (λ/D)")
      self.logger.debug(" -> " + sf(self.im_pscale, 2) + "\" per pixel (with γ=" + str(self.gamma) + " pixels per resolution element)")
      self.logger.debug(" -> a detector FoV of " + sf(self.detector_FOV, 2) + "\"")
      self.logger.debug(" -> an airy disk diameter of " + sf(self.airy_disk_d, 2) + "\"")
      
    # create mask  
    self._setup()
      
  def _setup(self):
    y, x = np.ogrid[-self.gsize/2:self.gsize/2, -self.gsize/2:self.gsize/2]
    mask = x*x + y*y <= (self.sampling/2)*(self.sampling/2)
    array = np.zeros((self.gsize, self.gsize))
    array[mask] = 1
    self.data = array
    
  def getData(self):
    return self.data
  
  def getScaledPupilDescriptors(self, n_airy_diameters):
    im_npix 			= int(np.ceil(self.gamma*2.44*n_airy_diameters))	# limit the number of pixels in the image
    extent_scale_factor		= float(self.getData().shape[0])/im_npix		# corresponding scale factor to scale extent
    detector_HFOV_scaled 	= (self.getDetectorHFOV())/extent_scale_factor		# HFOV of detector, scaled down by [scale]   
    if self.verbose:
      self.logger.debug(" Using innermost " + str(im_npix) + " pixels, giving a detector scaled HFOV of " + str(detector_HFOV_scaled) + "\".")
    return im_npix, detector_HFOV_scaled
	 
  def takeSlice(self, slice_width_in_resolution_elements):
    fractional_slice_width = int(round(1./slice_width_in_resolution_elements))
    if self.verbose:
      self.logger.debug(" Taking slice of width γ/" + str(fractional_slice_width) + " = " + str(self.gamma/2) + " pixels.")  
    self.data[0:(self.gsize/2)-(self.gamma/(2*fractional_slice_width))] = 0		# total width is half a resolution element, so need to divide by a further 2 if taking top/bottom limits
    self.data[(self.gsize/2)+(self.gamma/(2*fractional_slice_width)):] = 0   
    
    return fractional_slice_width
  
  def getExtent(self):
    ''' 
      Returns an extent with real units for image plots.
    '''
    if self.space == 'pupil':
      return (-self.physical_gsize/2, self.physical_gsize/2, -self.physical_gsize/2, self.physical_gsize/2)
    elif self.space == 'image':
      return (-self.getDetectorHFOV(), self.getDetectorHFOV(), -self.getDetectorHFOV(), self.getDetectorHFOV())
  
  def getDetectorHFOV(self):
    return self.detector_FOV/2
  
  def do2DFFT(self, method="fast", shift=False):
    if method == "fast":
      self.data = np.fft.fft2(self.data)
    if shift:
      self.data = np.fft.fftshift(self.data)  
    if self.space == "image":
      self.space = "pupil"
    else:
      self.space = "image"
    
  def getFFTAmplitude(self):
    return np.abs(self.data)
  
  def getFFTPhase(self):
    return np.angle(self.data)
  
  def addToPhase(self, p):
    mag = self.getFFTAmplitude()
    phase = self.getFFTPhase()+p
    
    re = mag * np.cos(phase)
    im = mag * np.sin(phase)
    
    self.data = re + 1j * im



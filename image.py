#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pylab as plt

from util import sf, resample2d
from fcomplex import *

class image(object):
  '''
    This class provides abstraction for all the fields and methods 
    that are non-specific to the geometry of a given image.
  '''
  def __init__(self, logger, pupil, i_data, wave, reimager, verbose):
    self.logger = logger
    self.pupil  = pupil
    self.wave   = float(wave)
    self.data   = i_data
    self.reimager = reimager

  def asRegion(self, slice_x_s, slice_x_e, slice_y_s, slice_y_e, verbose=False):
    '''
      Take a region defined by [slice_x_s], [slice_x_e], [slice_y_s], 
      [slice_y_e] and return a new image with the area outside of the 
      region zeroed.
    ''' 
    if verbose:
      self.logger.debug(" Taking region from x = " + str(slice_x_s) + " to " + \
        str(slice_x_e) + " and y = " + str(slice_y_s) + " to " + \
        str(slice_y_e) + " pixels")
      
    data = deepcopy(self.data)
    data[:,0:slice_x_s] = 0
    data[:,slice_x_e:] = 0
    data[0:slice_y_s,:] = 0
    data[slice_y_e:,:] = 0    
    
    return self.__class__(self.logger, self.pupil, data, self.wave, 
      self.reimager, verbose=False)    
  
  def getAmplitude(self, power=False, shift=False, scale="linear", 
                   normalise=False):
    return getAmplitude(self.logger, self.data, power=power, shift=shift, 
                 scale=scale, normalise=normalise)
  
  def getPhase(self, shift=False):
    return getPhase(self.logger, self.data, shift=shift)
  
  def getRealComponent(self, shift=False, normalise=False):
    return getRealComponent(self.logger, self.data, shift=shift, 
                            normalise=normalise)

  def getImagComponent(self, shift=False, normalise=False):
    return getImagComponent(self.logger, self.data, shift=shift, 
                            normalise=normalise)
    
  def resample(self, new_pixel_scale, verbose=True):
    '''
      Resample data to new pixel scale. This is an in-place operation.
    '''
    p_pixel_scale_micron = sf(self.p_pixel_scale*1e6, 4)
    p_new_pixel_scale_micron = sf(new_pixel_scale*1e6, 4)
    
    if np.isclose(self.p_pixel_scale, new_pixel_scale):
      self.logger.debug(" Resampling not required as pixel scales are " + \
                          "the same")
    else:
      if verbose:
        self.logger.debug(" Data needs to be resampled from " + \
          p_pixel_scale_micron + " to " + p_new_pixel_scale_micron + \
            " micron per pixel.")
      if np.issubdtype(self.data.dtype, np.complexfloating):
        if verbose:
          self.logger.debug(" Resampling complex data requires discarding " + \
            "the imaginary part. This only makes sense if the phase is zero.")
        if np.allclose(np.imag(self.data), 0):
          if verbose:
            self.logger.debug(" Checking to see if all imaginary parts are " + \
              "zero... OK")
        else:
          if verbose:
            self.logger.critical(" Checking to see if all imaginary parts " + \
              "are zero... FAIL")
          exit(0)

      self.data = np.real(self.data)
    
      # resample data
      new_FOV = new_pixel_scale * self.data.shape[0]
      self.data = resample2d(self.data, -self.p_detector_FOV/2., 
        self.p_detector_FOV/2., self.p_pixel_scale, -new_FOV/2., new_FOV/2., 
        new_pixel_scale)
        
      # convert back to complex
      self.data = self.data.astype(dtype=complex)

  def setRegionData(self, slice_x_s, slice_x_e, slice_y_s, slice_y_e, data):
    '''
      Set the data [self.data] in the region defined by [slice_x_s], 
      [slice_x_e], [slice_y_s], [slice_y_e] to [data].
    '''
    self.data[slice_y_s:slice_y_e, slice_x_s:slice_x_e] = \
    data[slice_y_s:slice_y_e, slice_x_s:slice_x_e]

  def toConjugatePupil(self, ishift=True, verbose=False):  
    '''
      Move back to conjugate pupil plane.
    '''
    from pupil import pupil
    if ishift:
      p_data = np.fft.ifftshift(self.data)   
    p_data = np.fft.ifft2(p_data)
    new_pupil = pupil_circular(self.pupil.logger, self.pupil.sampling, 
      self.pupil.gamma, verbose=False, data=p_data)
    return new_pupil    
    
class image_circular(image):
  '''
    Image class corresponding to image of a circular pupil.
  '''  
  def __init__(self, logger, pupil, i_data, wave, reimager, verbose):
    super(image_circular, self).__init__(logger, pupil, i_data, wave, reimager, 
                                         verbose)
    
    self.p_resolution_element  = reimager.getLinearResolutionElement(wave)
    self.p_pixel_scale         = reimager.getLinearPixelScale(wave, pupil)
    self.p_detector_FOV        = reimager.getLinearDetectorFOV(wave, pupil)
    self.p_airy_disk_d         = reimager.getLinearAiryDiskDiameter(wave)

    if verbose:
      logger.debug(" The image for a wavelength of " + \
        sf(self.wave*10**9, 4) + "nm and a reimager with a focal ratio of " + \
          sf(reimager.wfno, 3) + " has the following properties: ")
      self._printHumanReadableProperties()

  def _printHumanReadableProperties(self):
    a_resolution_element_asec = sf(np.degrees(
    self.pupil.getAngularResolutionElement(self.wave)*3600),4)
    p_resolution_element_micron = sf(self.p_resolution_element*1e6, 4)

    a_pixel_scale_asec = sf(np.degrees(
      self.pupil.getAngularPixelScale(self.wave)*3600), 4)
    p_pixel_scale_micron = sf(self.p_pixel_scale*1e6, 4)
    
    a_detector_FOV_asec = sf(np.degrees(
      self.pupil.getAngularDetectorFOV(self.wave)*3600), 4)
    p_detector_FOV_micron = sf(self.p_detector_FOV*1e6, 6)
    
    a_airy_disk_d_asec = sf(np.degrees(
      self.pupil.getAngularAiryDiskDiameter(self.wave)*3600), 4) 
    
    p_airy_disk_d_micron = sf(self.p_airy_disk_d*1e6, 4)

    self.logger.debug(" -> " + p_resolution_element_micron + " micron" + \
          " per resolution element")
    self.logger.debug(" -> " + p_pixel_scale_micron + " micron" + \
      " per pixel, with gamma=" + str(sf(self.pupil.gamma, 4)) + \
      " pixels per resolution element")
    self.logger.debug(" -> a FoV of " + p_detector_FOV_micron + " micron")
    self.logger.debug(" -> an airy disk diameter of " + \
      p_airy_disk_d_micron + " micron")
     
  def resample(self, new_pixel_scale, verbose=True):
    '''
        Resample data to new pixel scale. This is an in-place operation.
    '''
    super(image_circular, self).resample(new_pixel_scale, verbose)
  
    # change pupil and image parameters to reflect rescale
    self.pupil.gamma = self.p_resolution_element/new_pixel_scale
    self.pupil.physical_gsize = self.pupil.physical_pupil_diameter * \
      self.pupil.gamma
    self.pupil.sampling = self.pupil.gsize/self.pupil.gamma
    self.pupil.pupil_plate_scale = self.pupil.physical_gsize/self.pupil.gsize
    
    self.p_resolution_element = self.reimager.getLinearResolutionElement(
      self.wave)
    self.p_pixel_scale = self.reimager.getLinearPixelScale(self.wave, 
      self.pupil)
    self.p_detector_FOV = self.reimager.getLinearDetectorFOV(self.wave, 
      self.pupil)
    self.p_airy_disk_d = self.reimager.getLinearAiryDiskDiameter(self.wave)

    if verbose:
      self.logger.debug(" Image of circular pupil now has the following " + \
        "properties: ")
      self._printHumanReadableProperties()

  def toConjugatePupil(self, ishift=True, verbose=False):  
    '''
      Move back to conjugate pupil plane.
    '''
    from pupil import pupil_circular
    if ishift:
      p_data = np.fft.ifftshift(self.data)   
    p_data = np.fft.ifft2(p_data)
    new_pupil = pupil_circular(self.pupil.logger, self.pupil.sampling, 
      self.pupil.gamma, self.pupil.physical_pupil_radius, 
      verbose=False, data=p_data)
    return new_pupil          
      

    

      

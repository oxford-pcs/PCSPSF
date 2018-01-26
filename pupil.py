#!/usr/local/bin/python
# -*- coding: utf-8 -*-
from copy import deepcopy

import numpy as np

from util import sf
from fcomplex import *

class pupil(object):
  '''
    This class provides abstraction for all the fields and methods 
    that are non-specific to the geometry of a given pupil.
  '''
  def __init__(self, logger, sampling, gamma, verbose, data=None):
    self.logger = logger
    self.sampling = sampling # sampling size for pupil (pixels)
    self.gamma = gamma # factor by which to pad grid (=px/resolution element)
    
    if data is None: # if there's no data ..
      self.gsize = self.sampling*self.gamma # .. set size of grid in pixels
      self._setup() # .. set up a new data grid
    else:
      self.gsize = data.shape[0]
      self.data = data
    
  def _setup(self):
    '''
      Instantiate [self.data] with zero phase and unity magnitude across the
      array.
    '''
    mag = np.ones((self.gsize,self.gsize))
    phase = np.zeros(mag.shape)
    
    re = mag * np.cos(phase)
    im = mag * np.sin(phase)
    
    self.data = re + 1j * im
  
  def addToPhase(self, p):
    addToPhase(self.logger, self.data, p)
    
  def copy(self):
    return pupil(self.logger, self.sampling, self.gamma, verbose=False, 
                 data=self.data)
    
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
      
  def getExtent(self):
    ''' 
      Returns an extent with real units for image plots.
    '''
    return (-self.physical_gsize/2, self.physical_gsize/2, 
            -self.physical_gsize/2, self.physical_gsize/2)
    
  def toConjugateImage(self, wave, camera, shift=True, verbose=False):
    '''
      Move to conjugate image plane.
    '''
    from image import image
    i_data = np.fft.fft2(self.data)
    if shift:
      i_data = np.fft.fftshift(i_data)   
    return image(self.logger, self, i_data, wave, camera, verbose)
  
class pupil_circular(pupil): 
  '''
    Circular pupil class.
  '''
  def __init__(self, logger, sampling, gamma, physical_pupil_radius,
               verbose=False, data=None):
    super(pupil_circular, self).__init__(logger, sampling, gamma, verbose, data)
    self.physical_pupil_radius = physical_pupil_radius
    self.physical_pupil_diameter = self.physical_pupil_radius*2
    self.physical_gsize = self.physical_pupil_diameter*self.gamma  
    self.pupil_plate_scale = self.physical_gsize/self.gsize
    if verbose:
      self.logger.debug(" Entrance pupil diameter is " + \
        str(sf(self.physical_pupil_diameter*1e3, 3)) + 'mm')
      self.logger.debug(" With pupil sampling of " + str(sf(self.sampling, 4)) \
        + "x" + str(sf(self.sampling, 4)) + \
          ", this corresponds to a pupil plate scale of " + \
            sf(self.pupil_plate_scale*1e3, 2) + "mm/px")
   
  def _setup(self):
    '''
      Instantiate [self.data] with zero phase and unity magnitude for a circle 
      of radius [physical_pupil_radius] across the array.
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
    
  def copy(self):
    return pupil_circular(self.logger, self.sampling, self.gamma,
                    self.physical_pupil_radius, verbose=False, data=self.data)    
    
  def getAngularAiryDiskDiameter(self, wave):
    return 2.44*self.getAngularResolutionElement(wave)       # radians
  
  def getAngularPixelScale(self, wave):
    return self.getAngularResolutionElement(wave)/self.gamma # radians/px
  
  def getAngularResolutionElement(self, wave):
    return float(wave)/self.physical_pupil_diameter          # radians
  
  def getAngularDetectorFOV(self, wave):
    return self.getAngularPixelScale(wave)*self.gsize        # radians    
    
  def toConjugateImage(self, wave, camera, shift=True, verbose=False):
    '''
      Move to conjugate image plane.
    '''
    from image import image_circular
    i_data = np.fft.fft2(self.data)
    if shift:
      i_data = np.fft.fftshift(i_data)   
    return image_circular(self.logger, self, i_data, wave, camera, verbose)    
   


from copy import deepcopy

import numpy as np

def addToPhase(logger, data, p):
  '''
    Add to the phase of [data], and reform as a complex number.
  '''
  mag = self.getAmplitude()
  phase = self.getPhase()+p

  re = mag * np.cos(phase)
  im = mag * np.sin(phase)

  data = re + 1j * im
  return data

def getAmplitude(logger, data, power=False, shift=False, normalise=False, 
                scale="linear"):
  '''
    Get the amplitude of [data], or convert to a variety of formats.
  
    Returns a data array of the same shape as [data].
  '''
  d = deepcopy(np.abs(data))

  if power:
    d = d**2
  if scale != "linear":
    if scale == "log":
      d = np.log10(d)
    else:
      logger.warning(" Unrecognised scale keyword, assuming linear")
  if normalise:
    d = (d-np.min(d))/(np.max(d)-np.min(d))
  if shift:
    d = np.fft.fftshift(d) 
  return d

def getPhase(logger, data, shift=False):
  '''
    Get the phase of [data].
  
    Returns a data array of the same shape as [data].
  '''
  if shift:
    return np.angle(np.fft.fftshift(data))
  return np.angle(data)

def getRealComponent(logger, data, shift=False, normalise=False):
  '''
    Returns the real component of [data].
  '''
  d = deepcopy(data)
  if shift:
    d = np.fft.fftshift(d)
  re = np.real(d)
  if normalise:
    re = (re-np.min(re))/(np.max(re)-np.min(re))
  return re
  
def getImagComponent(logger, data, shift=False, normalise=False):
  '''
    Returns the imaginary component of [data].
  '''
  d = deepcopy(data)
  if shift:
    d = np.fft.fftshift(d)
  im = np.imag(d)
  if normalise:
    im = (im-np.min(im))/(np.max(im)-np.min(im))
  return im    

#!/usr/local/bin/python
# -*- coding: utf-8 -*-

'''
  mwe.py
  
  DESCRIPTION
  
  This program is a minimal working example, illustrating the proof of concept for the psf_simulator.
  
  NOTES
 
  EXAMPLES
  
  $ python mwe.py
'''
import logging
import os

import numpy as np
import pyfits
import sys
import pylab as plt
import pyfits

from zmx_parser import zwfe
from util import resample2d

wsize = 2048		# wfe map resampled size

gsize = 2048		# grid size
sampling = 256		# size of pupil
shalfwidth = 2		# slice half width
vhalfwidth = 10		# display half width

range_y = [(gsize/2)-vhalfwidth, (gsize/2)+vhalfwidth+1]
range_x = [(gsize/2)-vhalfwidth, (gsize/2)+vhalfwidth+1]

def run(wfe_map, max_wfe, plot):
  
  # Construct pupil
  y, x = np.ogrid[-gsize/2:gsize/2, -gsize/2:gsize/2]
  mask = x*x + y*y <= (sampling/2)*(sampling/2)
  mag = np.zeros((gsize, gsize))
  mag[mask] = 1
      
  phase = np.zeros(mag.shape)
      
  re = mag * np.cos(phase)
  im = mag * np.sin(phase)
      
  pupil = re + 1j * im
 
  # Move pupil to non-dc-centered
  pupil = np.fft.fftshift(pupil)
 
  # Move to image space
  image = np.fft.fft2(pupil)
  
  # Move image to dc-centered for slicing	
  image = np.fft.fftshift(image)	
  
  # take a slice of the power distribution of the image, this is the intensity distribution we would see at the slicer
  image_pow = np.abs(image)**2
  slice_range_y = (int((image.shape[0]/2)-shalfwidth),int((image.shape[0]/2)+shalfwidth)+1)
  slice_range_x = (0, image.shape[1])
  image_slice_pow = np.zeros(shape=image.shape)
  image_slice_pow[slice_range_y[0]:slice_range_y[1], slice_range_x[0]:slice_range_x[1]] = image_pow[slice_range_y[0]:slice_range_y[1], slice_range_x[0]:slice_range_x[1]]
  
  # normalise for convenience 
  image_slice_pow /= np.max(image_slice_pow)

  # construct complex array for sliced image with power as magnitude and phase = 0
  phase = np.zeros(image_slice_pow.shape)      
  re = image_slice_pow * np.cos(phase)
  im = image_slice_pow * np.sin(phase)
  image_slice = re + 1j * im

  image_slice_pow = np.abs(image_slice)**2

  # Move image to non-dc-centered for FFT
  image_slice = np.fft.fftshift(image_slice)

  # Move to pupil space
  pupil_slice = np.fft.fft2(image_slice)
  
  mag = 1.0
  z = zwfe(wfe_map, logger)
  z.parse()
  w = z.getData(20)
  w = np.fft.fftshift(w)
  w = resample2d(w, 0, z.getHeader()['SAMPLING'][0], 1, 0, z.getHeader()['SAMPLING'][0], float(z.getHeader()['SAMPLING'][0])/float(wsize), kx=3, ky=3, s=0, gauss_sig=9, clip=True)
  
  diff = (image_slice.shape[0]-w.shape[0])/2
  if diff > 0:
    w = np.pad(w, diff, 'constant')
  elif diff < 0:
    w = w[diff+w.shape[0]/2:-diff+w.shape[0]/2,diff+w.shape[0]/2:-diff+w.shape[0]/2]
    
  # There are two options to add the WFE map, either by convolution (or rather 
  # multiplication in fourier space) or by adding the phase (essentially
  # treating the error as a phase-shifting plate.
  #
  # i) convolution
  if max_wfe == -1:
    phase = w
  else:
    phase = (w/np.max(w))*max_wfe
   
  re = mag * np.cos(phase)
  im = mag * np.sin(phase)
  w_p = re + 1j * im
  
  # Move to dc-centered and convolve wfe in fourier space
  pupil_slice = np.fft.fftshift(pupil_slice)
  pupil_slice_w = w_p*pupil_slice
  
  # Move back to non-dc-centered
  pupil_slice = np.fft.ifftshift(pupil_slice)
  pupil_slice_w = np.fft.ifftshift(pupil_slice_w)
  
  pyfits.writeto("w.fits", np.angle(w_p))
  exit(0)
   
  
  # ii) phase addition
  # Move to dc-centered and add phase
  '''
  pupil_slice_w = np.fft.fftshift(pupil_slice)  

  mag = np.abs(pupil_slice_w)
  phase = np.angle(pupil_slice_w)
  
  re = mag * np.cos(w+phase)
  im = mag * np.sin(w+phase)
  
  re = mag * np.cos(np.zeros(shape=mag.shape))
  im = mag * np.sin(np.zeros(shape=mag.shape))
  
  pupil_slice_w = re + 1j * im  
  
  # Move back to non-dc-centered
  pupil_slice_w = np.fft.ifftshift(pupil_slice_w)
  '''

  # Make corresponding slices in image space and dc-center
  image_slice_processed = np.fft.ifft2(pupil_slice)
  image_slice_processed = np.fft.fftshift(image_slice_processed)	

  image_slice_processed_w = np.fft.ifft2(pupil_slice_w)
  image_slice_processed_w = np.fft.fftshift(image_slice_processed_w)
  #image_slice_processed_w[0:(image_slice_processed_w.shape[1]/2)-shalfwidth,:] = 0 
  #image_slice_processed_w[(image_slice_processed_w.shape[1]/2)+shalfwidth:,:] = 0
  
  d = np.abs(image_slice_processed)**2			# slice w/ no wfe
  dp = np.abs(image_slice_processed_w)**2		# slice w/ wfe
  
  if plot:
    plt.subplot(441)
    plt.title("pupil mag [1]")
    plt.imshow(np.fft.fftshift(np.abs(pupil)), interpolation='none')
    plt.colorbar()
    
    plt.subplot(442)
    plt.title("pupil phase [2]")
    plt.imshow(np.fft.fftshift(np.angle(pupil)), interpolation='none')
    plt.colorbar()

    plt.subplot(443)
    plt.title("image power [3]")
    plt.imshow(image_pow/np.sqrt(image_pow.size), interpolation='none')
    plt.xlim(range_x)
    plt.ylim(range_y)
    plt.colorbar()
    
    plt.subplot(444)
    plt.title("normalised image slice power [4]")
    plt.imshow(image_slice_pow, interpolation='none')
    plt.xlim(range_x)
    plt.ylim(range_y)
    plt.colorbar()

    plt.subplot(445)
    plt.title("pupil mag after slicing [5]")
    plt.imshow(np.abs(np.fft.fftshift(pupil_slice)), interpolation='none')
    plt.colorbar()
    
    plt.subplot(446)
    plt.title("pupil phase after slicing [6]")
    plt.imshow(np.angle(np.fft.fftshift(pupil_slice)), interpolation='none')
    plt.colorbar()
    
    '''plt.subplot(447)
    plt.title("wfe map mag [7]")
    plt.imshow(np.abs(w_p), interpolation='none', vmax=1, vmin=0)
    plt.colorbar()
    
    plt.subplot(448)
    plt.title("wfe map phase [8]")
    plt.imshow(np.angle(w_p), interpolation='none')
    plt.colorbar()'''
    
    plt.subplot(4,4,9)  
    plt.title("sliced pupil mag w/ wfe [9]")
    plt.imshow(np.abs(np.fft.ifftshift(pupil_slice_w)), interpolation='none')
    plt.colorbar()
    
    plt.subplot(4,4,10)  
    plt.title("sliced pupil phase w/ wfe [10]")
    plt.imshow(np.angle(np.fft.ifftshift(pupil_slice_w)), interpolation='none')
    plt.colorbar()
   
    plt.subplot(4,4,11)   
    plt.title("image no wfe (" + str(np.max(d)) + ") [11]")
    plt.imshow(d, interpolation='none')
    plt.colorbar()
    plt.xlim(range_x)
    plt.ylim(range_y)

    plt.subplot(4,4,12)
    plt.title("image wfe (" + str(np.max(dp)) + ") [12]")
    plt.imshow(dp, interpolation='none')
    plt.colorbar()
    plt.xlim(range_x)
    plt.ylim(range_y)
    
    plt.subplot(4,4,13)  
    plt.title("image w/wfe x profile [13]")
    plt.plot(d[dp.shape[1]/2], label="nowfe")
    plt.plot(dp[dp.shape[1]/2], label="wfe")
    plt.xlim((0,2048))
    plt.legend()
    
    plt.subplot(4,4,14)  
    plt.title("image w/wfe y profile [14]")
    plt.plot(np.swapaxes(d, 0, 1)[dp.shape[1]/2], label="nowfe")
    plt.plot(np.swapaxes(dp, 0, 1)[dp.shape[1]/2], label="wfe")
    plt.xlim((0,2048))
    plt.legend()
    
    plt.subplot(4,4,15)  
    plt.title("wfe map (from file) [15]")
    plt.imshow(w)
    plt.colorbar()
    
    plt.show()
    
  return d, dp, np.abs(w)**2
    
if __name__== "__main__":
  # Setup logger
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  formatter = logging.Formatter("(" + str(os.getpid()) + ") %(asctime)s:%(levelname)s: %(message)s")
  ch.setFormatter(formatter)
  logger.addHandler(ch)
  
  d, dp, w = run("/local/home/barnsley/metadata/1/WFE_CAM_0_4", -1, True)

  '''DP = []  
  for i in np.arange(0,13,4):
    print i
    d, dp = run("/local/home/barnsley/metadata/1/WFE_CAM_0_14", i, False)
    
    DP.append(dp[range_y[0]:range_y[1],range_x[0]:range_x[1]])
  if not os.path.exists("out.fits"):
    pyfits.writeto("out.fits", np.array(DP))
  
  for idx, i in enumerate(DP):
    plt.plot(np.mean(i, axis=1), label=str(idx))
  plt.legend()
  plt.show()
  exit(0)'''
  
  '''D = []
  DP = []  
  W = []
  for i in np.arange(0,14,1):
    print i
    d, dp, w = run("/local/home/barnsley/metadata/1/WFE_CAM_0_" + str(i), -1, False)
    
    D.append(d[range_y[0]:range_y[1],range_x[0]:range_x[1]])
    DP.append(dp[range_y[0]:range_y[1],range_x[0]:range_x[1]])
    W.append(w)

  pyfits.writeto("D.fits", np.array(D))
  pyfits.writeto("DP.fits", np.array(DP))
  pyfits.writeto("W.fits", np.array(W))'''
 
  
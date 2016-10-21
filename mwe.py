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
import astropy.stats

from  PyQt4.QtGui import QApplication

from zmx_parser import zwfe
from util import resample2d
from ui import ui
from editable import editable

oversample = 8			# grid size
sampling = 256			# size of pupil
shalfwidth = 2			# slice half width
vhalfwidth = 20			# display half width

wsize = sampling*oversample	# wfe map resampled size

range_y = [((oversample*sampling)/2)-vhalfwidth, ((oversample*sampling)/2)+vhalfwidth+1]
range_x = [((oversample*sampling)/2)-vhalfwidth, ((oversample*sampling)/2)+vhalfwidth+1]

def run(wfe_map, max_wfe, plot, hard):
  
  app = QApplication(sys.argv)		# start app
 
  # Construct pupil
  y, x = np.ogrid[-sampling/2:sampling/2, -sampling/2:sampling/2]
  mask = x*x + y*y <= ((sampling-1)/2)*((sampling-1)/2)
  mag = np.zeros((sampling, sampling))
  mag[mask] = 1
      
  phase = np.zeros(mag.shape)
      
  re = mag * np.cos(phase)
  im = mag * np.sin(phase)
      
  pupil = re + 1j * im
  
  if hard:
    pyfits.writeto("pupil.fits", (np.abs(pupil)**2)/(np.max(np.abs(pupil)**2)))
  
  # oversize pupil
  pupil = np.pad(pupil, (sampling*(oversample-1))/2, mode='constant')  
  
  # edit pupil
  pui = ui()			# create ui instance
  e = editable(pui, pupil)	# create editable instance
  pupil = e.go()		# connect events and exec app
  
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
  
  # Construct map of WFE magnitude (truncated to width of diffracted slice)
  y, x = np.ogrid[-wsize/2:wsize/2, -wsize/2:wsize/2]
  mask = x*x + y*y <= (((wsize))/2)*(((wsize))/2)
  mag = np.zeros(((wsize), (wsize)))
  mag[mask] = 1
  
  d = np.abs(np.fft.fftshift(pupil_slice))**2
  d[np.isclose(d, 0, atol=1E-3)] = 0
  d_m = np.median(d, axis=0)
  lo = np.min(np.where(d_m>0))
  hi = np.max(np.where(d_m>0))
  
  mag[:,0:lo] = 0
  mag[:,hi:] = 0
  
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
    phase = np.zeros(shape=mag.shape)
    phase_w = w
  else:
    phase = np.zeros(shape=mag.shape)
    phase_w = (w/np.max(w))*max_wfe
   
  re = mag * np.cos(phase)
  im = mag * np.sin(phase)
  p = re + 1j * im   
   
  re = mag * np.cos(phase_w)
  im = mag * np.sin(phase_w)
  w_p = re + 1j * im
 
  # Move to dc-centered and convolve wfe in fourier space
  pupil_slice = np.fft.fftshift(pupil_slice)
  pupil_slice_w = pupil_slice*w_p
  pupil_slice = pupil_slice*p

  # Move back to non-dc-centered
  pupil_slice = np.fft.ifftshift(pupil_slice)
  pupil_slice_w = np.fft.ifftshift(pupil_slice_w)
  
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
  
  if hard:
    pyfits.writeto("spupil.fits", (np.abs(pupil_slice)**2)/(np.max(np.abs(pupil_slice)**2)))   
    pyfits.writeto("spupil_w.fits", (np.abs(pupil_slice_w)**2)/(np.max(np.abs(pupil_slice_w)**2)))    
    pyfits.writeto("nophase.fits", np.zeros(shape=pupil.shape))
    pyfits.writeto("w.fits", np.angle(w_p))
      
  # Make corresponding slices in image space and dc-center
  image_slice_processed = np.fft.ifft2(pupil_slice)
  image_slice_processed = np.fft.fftshift(image_slice_processed)

  image_slice_processed_w = np.fft.ifft2(pupil_slice_w)
  image_slice_processed_w = np.fft.fftshift(image_slice_processed_w)
  #image_slice_processed_w[0:(image_slice_processed_w.shape[1]/2)-shalfwidth,:] = 0 
  #image_slice_processed_w[(image_slice_processed_w.shape[1]/2)+shalfwidth:,:] = 0
  
  d = np.abs(image_slice_processed)**2			# slice w/ no wfe
  dp = np.abs(image_slice_processed_w)**2		# slice w/ wfe
  
  if hard:
    pyfits.writeto("out.fits", dp)
  
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
    
    plt.imshow(phase, interpolation='none')
    plt.colorbar()

    plt.subplot(447)
    plt.title("wfe map mag [7]")
    plt.imshow(np.abs(w_p), interpolation='none', vmax=1, vmin=0)
    plt.colorbar()
    
    plt.subplot(448)
    plt.title("wfe map phase [8]")
    plt.imshow(np.angle(w_p), interpolation='none')
    plt.colorbar()
    
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
    
    phase = np.angle(np.fft.fftshift(pupil_slice))
    power = np.abs(np.fft.fftshift(pupil_slice))**2
    rms_intensity_weighted_phase = np.sqrt((np.sum(power*(phase**2)))/len(phase))/np.sqrt(np.sum(power)/len(phase))
    print rms_intensity_weighted_phase/(2*np.pi)
    
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
  
  d, dp, w = run("/local/home/barnsley/metadata/1/WFE_CAM_0_4", -1, True, False)
  
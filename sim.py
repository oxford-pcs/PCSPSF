#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import random
import plotter
from pupil import circle
import ConfigParser
import logging
import copy
from zmx_parser import wf

def take2DFFT(im, method="fast", shift=True):
  ''' 
    returns fft, amplitude, phase and power
  '''
  if method == "fast":
    im_fft = np.fft.fft2(im)
  if shift:
    im_fft = np.fft.fftshift(im_fft)
  return im_fft, np.abs(im_fft), np.abs(im_fft)**2, np.angle(im_fft)

if __name__== "__main__":
  
  # Setup logger and plotting instance
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  formatter = logging.Formatter("%(levelname)s:%(asctime)s:%(message)s")
  ch.setFormatter(formatter)
  logger.addHandler(ch)
  
  pl = plotter.plotter()

  # Get relevant configuration parameters
  cfg = ConfigParser.ConfigParser()
  cfg.read("default.ini")
  PUPIL_RAD 	= float(cfg.get("general", "pupil_radius"))		
  PUPIL_GAM 	= int(cfg.get("general", "pupil_gamma"))
  LAMBDA	= float(cfg.get("general", "lambda"))
  GRID_UNIT	= str(cfg.get("general", "grid_unit"))

  pupil_diameter	= PUPIL_RAD*2						# metres
  pupil_osize 		= pupil_diameter*PUPIL_GAM				# metres
  im_resolution		= np.degrees(LAMBDA/pupil_diameter)*3600		# "/resolution element
  im_pscale		= im_resolution/PUPIL_GAM				# "/px
  
  if GRID_UNIT == "mm":
    pupil_mfactor	= 1000
    
  pupil_diameter	= pupil_diameter*pupil_mfactor
  pupil_osize 		= pupil_osize*pupil_mfactor
  detector_FOV 		= im_pscale*pupil_osize

  logger.debug(" Pupil diameter is " + str(pupil_diameter) + GRID_UNIT)
  logger.debug(" Pupil plate scale is " + GRID_UNIT + "/px.")
  logger.debug(" γ=" + str(PUPIL_GAM) + " (pixels per unit of angular resolution).")
  logger.debug(" At a wavelength of " + str(LAMBDA*10**9) + "nm, this corresponds to:")
  logger.debug(" - an image plate scale of " + str(im_pscale) + "\"/px.")
  logger.debug(" - a detector FoV of " + str(detector_FOV) + "\".")
  
  # construct pupil
  pupil = circle(pupil_osize/2, pupil_osize/2, pupil_osize, pupil_diameter/2)
  pl._addImagePlot("pupil", pupil)

  # fft and shift to move from pupil to image space
  pupil_fft_shift, pupil_fft_shift_A, pupil_fft_shift_pow, pupil_fft_shift_p = take2DFFT(pupil, method="fast")
  pl._addImagePlot("-> fft (amplitude)", pupil_fft_shift_A, extent=(-detector_FOV/2,detector_FOV/2,-detector_FOV/2,detector_FOV/2), xl="arcsec", yl="arcsec")
  
  # take a slice corresponding to half a resolution element
  logger.debug(" Taking slice of width γ/2 = " + str(PUPIL_GAM/2) + " pixels.")  
  pupil_fft_shift_slice = copy.deepcopy(pupil_fft_shift)
  pupil_fft_shift_slice[0:(pupil_osize/2)-(PUPIL_GAM/2)] = 0
  pupil_fft_shift_slice[(pupil_osize/2)+(PUPIL_GAM/2):] = 0

  # fft back to pupil plane
  pupil_fft_shift_slice_fft, pupil_fft_shift_slice_fft_A, pupil_fft_shift_slice_fft_pow, pupil_fft_shift_slice_fft_p = take2DFFT(pupil_fft_shift_slice, method="fast", shift=False)
  pl._addImagePlot("-> slice -> fft", pupil_fft_shift_slice_fft_A)
  
  # add WFE
  wf1 = wf("DEFAULT.TXT", logger)
  wf1.parse()
  wfe = wf1.getData()
  wfe_p = wfe*2*np.pi
  pl._addImagePlot("wavefront phase error", wfe_p)
  
  pl.draw(2,2)








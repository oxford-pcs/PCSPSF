#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import random
import logging
import copy
import ConfigParser
import os
import sys
import argparse
import time

import numpy as np
import pyfits
import pyds9

import plotter
from pupil import circular
from camera import camera
from zmx_parser import zwfe
from util import sf

def isPowerOfTwo(num):
  while num % 2 == 0 and num > 1:
    num = num/2
  return num == 1

class sim():
  def __init__(self, logger, plotter, cfg_file):
    self.logger 	= logger
    self.plotter 	= plotter
    self.cfg_file	= cfg_file

  def run(self, wave, wfe_file, plot=True, fits=True, ds9=False, verbose=True):
    """
      Run the simulation.
    """

    # get configuration parameters
    cfg = ConfigParser.ConfigParser()
    cfg.read(self.cfg_file)
    CAMERA_FWNO			= float(cfg.get("camera", "wfno"))
    
    PUPIL_SAMPLING		= float(cfg.get("pupil", "pupil_sampling"))
    PUPIL_GAMMA 		= float(cfg.get("pupil", "pupil_gamma"))
    PUPIL_RADIUS 		= float(cfg.get("pupil", "pupil_radius_physical"))
    PUPIL_RADIUS_UNIT		= str(cfg.get("pupil", "pupil_radius_physical_unit"))
    
    ADD_WFE			= bool(int(cfg.get("wfe", "do")))

    DO_SLICING			= bool(int(cfg.get("slicing", "do")))
    NSLICES			= int(cfg.get("slicing", "number"))
    SLICE_WIDTH			= float(cfg.get("slicing", "width"))				# in resolution elements

    try:
      PUPIL_SAMPLING 	= int(PUPIL_SAMPLING)
      PUPIL_GAMMA 	= int(PUPIL_GAMMA)
    except ValueError:
      self.logger.critical(" PUPIL_SAMPLING and PUPIL_GAMMA should be an integer!")
      exit(0)
      
    try:
      assert PUPIL_GAMMA % 2 == 0
    except AssertionError:
      self.logger.warning(" Pupil gamma should be even. Could produce unexpected results.")    
      
    try:
      assert NSLICES % 2 == 1
    except AssertionError:
      self.logger.critical(" Number of slices should be odd!")
      exit(0)
      
    try:
      assert isPowerOfTwo(PUPIL_SAMPLING) == True
    except AssertionError:
      self.logger.critical(" Pupil sampling should be a power of two!")
      exit(0)
      
    # instantiate camera and entrance pupil
    cam = camera(CAMERA_FWNO)
    pupil = circular(self.logger, cam, PUPIL_SAMPLING, PUPIL_GAMMA, PUPIL_RADIUS, PUPIL_RADIUS_UNIT, verbose=verbose)      
      
    # get WFE if requested
    if ADD_WFE:
      wfe = zwfe(wfe_file, logger, verbose=verbose)
      wfe.parse()
      wfe_h = wfe.getHeader()	
      wfe_d = wfe.getData(in_radians=True)		# returns data same dimensions as pupil array, in radians
	  
      if wfe_h['SAMPLING'][0] != PUPIL_SAMPLING:
	self.logger.critical(" Zemax WFE sampling is not the same as the pupil sampling! (" + str(wfe_h['SAMPLING'][0]) + " != " + str(PUPIL_SAMPLING) + ")")
	exit(0)
      pad_by = (PUPIL_SAMPLING*(PUPIL_GAMMA-1))/2
      wfe_d = np.pad(wfe_d, pad_by, mode='constant')
      pl.addImagePlot("wfe (radians)", np.abs(wfe_d), extent=pupil.getExtent(), xl=PUPIL_RADIUS_UNIT, yl=PUPIL_RADIUS_UNIT)  

    pl.addImagePlot("pupil", pupil.getAmplitude(), extent=pupil.getExtent(), xl=PUPIL_RADIUS_UNIT, yl=PUPIL_RADIUS_UNIT)

    # plot image of whole pupil unsliced (this image is just for plotting purposes)
    im = pupil.toConjugateImage(wave)
    d, hfov = im.getAmplitudeScaledByAiryDiameters(3, scale="linear", normalise=True)
    pl.addImagePlot("-> fft to image space", d, extent=(-hfov, hfov, -hfov, hfov), xl="arcsec", yl="arcsec")

    # slicing
    if DO_SLICING:
      # take slices from the image space
      slices = []
      for s in range(NSLICES):				
	im = pupil.toConjugateImage(wave)		# move from pupil to image space
	offset = (s-((NSLICES-1)/2))*SLICE_WIDTH
	im.takeSlice(SLICE_WIDTH, offset=offset)	# create a new pupil conjugate image instance for each slice
	pl.addScatterPlot(None, [(-(SLICE_WIDTH*im.resolution_element)/2)+(offset*im.resolution_element), 
				(-(SLICE_WIDTH*im.resolution_element)/2)+(offset*im.resolution_element)], [-hfov, hfov], xr=(-hfov, hfov), yr=(-hfov, hfov), overplot=True)
	pl.addScatterPlot(None, [((SLICE_WIDTH*im.resolution_element)/2)+(offset*im.resolution_element), 
				((SLICE_WIDTH*im.resolution_element)/2)+(offset*im.resolution_element)], [-hfov, hfov], xr=(-hfov, hfov), yr=(-hfov, hfov), overplot=True)
	pl.addTextToPlot(hfov-(hfov/4), (offset*im.resolution_element)-((SLICE_WIDTH*im.resolution_element)/2), str(s), color='w', fontsize=10)
	slices.append(im)
	
      for idx, s in enumerate(slices):	
	 # fft back to pupil plane 
	new_pupil = s.toConjugatePupil()				
        pl.addImagePlot("-> take slice " + str(idx) + " -> ifft to pupil space", new_pupil.getAmplitude(), extent=new_pupil.getExtent(), xl=PUPIL_RADIUS_UNIT, yl=PUPIL_RADIUS_UNIT)
    	
        # add phase WFE 
        if ADD_WFE:
	  new_pupil.addToPhase(wfe_d)
	  plt_title_prefix = "added phase error "
	else:
	  plt_title_prefix = ""
    
        # fft back to image plane
        im = new_pupil.toConjugateImage(wave)
        d, hfov = im.getAmplitudeScaledByAiryDiameters(3, shift=True)
        pl.addImagePlot(plt_title_prefix + "-> fft to image space", d, extent=(-hfov, hfov, -hfov, hfov), xl="arcsec", yl="arcsec")
    else:
      # move from pupil to image space
      im = pupil.toConjugateImage(wave)
      # add phase WFE 
      if ADD_WFE:
        pupil.addToPhase(wfe_d)
        plt_title_prefix = "added phase error "
      else:
	  plt_title_prefix = ""
	  
      d, hfov = im.getAmplitudeScaledByAiryDiameters(20, scale="log", normalise=True)
      pl.addImagePlot(plt_title_prefix + "-> fft to image space", d, extent=(-hfov, hfov, -hfov, hfov), xl="arcsec", yl="arcsec")
    
    if plot:
      pl.draw(3,4)
    
    if fits:
      if os.path.exists("out.fits"):
        os.remove("out.fits")
      h = pyfits.Header()
      h.append(('CRVAL1', -im.getDetectorHFOV()))
      h.append(('CDELT1', im.pscale))
      h.append(('CRPIX1', 0.5))
      h.append(('CUNIT1', "arcsec"))
      h.append(('CTYPE1', "PARAM"))
      h.append(('CRVAL2', -im.getDetectorHFOV()))
      h.append(('CDELT2', im.pscale))
      h.append(('CRPIX2', 0.5))
      h.append(('CUNIT2', "arcsec"))
      h.append(('CTYPE2', "PARAM"))
      pyfits.writeto("out.fits", im.getAmplitude(power=True, shift=False, normalise=True), h)
      
      if ds9:
	d = pyds9.DS9()
	d.set("file out.fits")
	d.set('cmap heat')
	d.set('scale log')
	d.set('zoom 4')

if __name__== "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-ws", help="wavelength start", default=800e-9, type=float)
  parser.add_argument("-we", help="wavelength end", default=801e-9, type=float)
  parser.add_argument("-wi", help="wavelength interval", default=25e-9, type=float)
  parser.add_argument("-e", help="zemax wfe file directory", default="etc/wfe/diffg", type=str)	# should be in format 
  parser.add_argument("-p", help="plot?", action="store_true")
  parser.add_argument("-f", help="create fits", action="store_true")
  parser.add_argument("-fv", help="view fits", action="store_true")
  parser.add_argument("-v", help="verbose", action="store_true")
  args = parser.parse_args()

  # setup logger
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  formatter = logging.Formatter("%(levelname)s:%(asctime)s:%(message)s")
  ch.setFormatter(formatter)
  logger.addHandler(ch)
  
  pl = plotter.plotter()	# setup plotting
  
  logger.debug(" Starting timer.")
  st = time.time()

  s = sim(logger, plotter, "etc/default.ini")
  for w in np.arange(args.ws, args.we, args.wi):
    logger.info(" Beginning simulation for a wavelength of " + str(w*1e9) + "nm...")   
    
    # find appropriate zemax wfe file
    wfe_file = None
    for f in os.listdir(args.e):
      f = args.e.rstrip('/') + '/' + f
      wfe = zwfe(f, logger, verbose=False)
      if wfe.parseFileHeader():
        h = wfe.getHeader()
        if np.isclose(h['WAVE']*h['WAVE_EXP'], w):
	  wfe_file = f
	  break
    if wfe_file is None:
      logger.critical(" Unable to find Zemax WFE file!")
      exit(0)
 
    s.run(w, wfe_file, plot=args.p, fits=args.f, ds9=args.fv, verbose=args.v)
    
  fi = time.time()
  duration = fi-st
  logger.debug(" Ending timer.")
  logger.debug(" Full simulation completed in " + str(sf(duration, 4)) + "s.")

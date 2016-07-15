#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import random
import logging
import copy
import ConfigParser
import os

import numpy as np
import pyfits
import pyds9

import plotter
from pupil import circular
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

  def run(self, wave, wfe_file, plot=True, ds9=True, verbose=True):
    """
      Run the simulation.
    """

    # get configuration parameters
    cfg = ConfigParser.ConfigParser()
    cfg.read(self.cfg_file)
    PUPIL_SAMPLING		= float(cfg.get("general", "pupil_sampling"))
    PUPIL_GAMMA 		= float(cfg.get("general", "pupil_gamma"))
    PUPIL_RADIUS 		= float(cfg.get("general", "pupil_radius_physical"))
    PUPIL_RADIUS_UNIT		= str(cfg.get("general", "pupil_radius_physical_unit"))
    
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
      self.logger.warning(" Pupil gamma should be even.")    
      
    try:
      assert isPowerOfTwo(PUPIL_SAMPLING) == True
    except AssertionError:
      self.logger.critical(" Pupil sampling should be a power of two!")
      exit(0)
      
    # construct entrance pupil
    pupil = circular(self.logger, PUPIL_SAMPLING, PUPIL_GAMMA, PUPIL_RADIUS, PUPIL_RADIUS_UNIT, verbose=verbose)
    pl.addImagePlot("pupil", pupil.getAmplitude(), extent=pupil.getExtent(), xl=PUPIL_RADIUS_UNIT, yl=PUPIL_RADIUS_UNIT)
    
    # move from pupil to image space
    im = pupil.toConjugateImage(801e-9)
    d, hfov = im.getAmplitudeScaledByAiryDiameters(3)
    pl.addImagePlot("-> fft to image space (A)", d, extent=(-hfov, hfov, -hfov, hfov), xl="arcsec", yl="arcsec")
    
    # take a slice from the image space
    im.takeSlice(SLICE_WIDTH)
    pl.addScatterPlot(None, [-(SLICE_WIDTH*im.im_resolution_element)/2, -(SLICE_WIDTH*im.im_resolution_element)/2], [-hfov, hfov], xr=(-hfov, hfov), yr=(-hfov, hfov), overplot=True)
    pl.addScatterPlot(None, [(SLICE_WIDTH*im.im_resolution_element)/2, (SLICE_WIDTH*im.im_resolution_element)/2], [-hfov, hfov], overplot=True)
        
    # fft back to pupil plane
    im.toConjugatePupil()
    pl.addImagePlot("-> take slice -> ifft to pupil space", pupil.getAmplitude(), extent=pupil.getExtent(), xl=PUPIL_RADIUS_UNIT, yl=PUPIL_RADIUS_UNIT)
    
    # add WFE
    wfe = zwfe(wfe_file, logger)
    wfe.parse()
    wfe_h = wfe.getHeader()
    wfe_d = wfe.getData(match_pupil=pupil)	# returns data same dimensions as pupil array, in radians
    
    pl.addImagePlot("wfe (radians)", np.abs(wfe_d), extent=pupil.getExtent(), xl=PUPIL_RADIUS_UNIT, yl=PUPIL_RADIUS_UNIT)	
    pupil.addToPhase(wfe_d)
    
    # fft back to image plane
    im = pupil.toConjugateImage(801e-9, shift=False)
    d, hfov = im.getAmplitudeScaledByAiryDiameters(5)

    pl.addImagePlot("-> fft to image space (A)", d, extent=(-hfov, hfov, -hfov, hfov), xl="arcsec", yl="arcsec")
 
    if plot:
      pl.draw(3,2)
    
    if ds9:
      pyfits.writeto("out.fits", d)
      d = pyds9.DS9()
      d.set("file out.fits")
      os.remove("out.fits")

if __name__== "__main__":
  
  # setup logger
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  formatter = logging.Formatter("%(levelname)s:%(asctime)s:%(message)s")
  ch.setFormatter(formatter)
  logger.addHandler(ch)
  
  pl = plotter.plotter()	# setup plotting

  s = sim(logger, plotter, "etc/default.ini")
  s.run(801e-9, "etc/801_0_0_256x256", plot=True, ds9=False)

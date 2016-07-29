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
import pyds9

import plotter
from pupil import circular
from camera import camera
from zmx_parser import zwfe
from products import cube
from util import sf

def isPowerOfTwo(num):
  while num % 2 == 0 and num > 1:
    num = num/2
  return num == 1

class sim():
  def __init__(self, logger, plotter, resampling_im, nwaves, CAMERA_FWNO, PUPIL_SAMPLING, 
	       PUPIL_GAMMA, PUPIL_RADIUS, ADD_WFE, DO_SLICING, 
	       NSLICES, SLICE_WIDTH, RESAMPLE_TO):
    self.logger 		= logger
    self.plotter 		= plotter
    self.resampling_im		= resampling_im
    self.nwaves			= nwaves
    self.CAMERA_FWNO		= CAMERA_FWNO
    self.PUPIL_SAMPLING 	= PUPIL_SAMPLING
    self.PUPIL_GAMMA		= PUPIL_GAMMA
    self.PUPIL_RADIUS		= PUPIL_RADIUS
    self.ADD_WFE		= ADD_WFE
    self.DO_SLICING		= DO_SLICING
    self.NSLICES		= NSLICES
    self.SLICE_WIDTH		= SLICE_WIDTH
    self.RESAMPLE_TO		= RESAMPLE_TO
    
    self.datacube		= cube(self.logger, shape=[self.nwaves, self.PUPIL_SAMPLING*self.PUPIL_GAMMA, self.PUPIL_SAMPLING*self.PUPIL_GAMMA])
   
  def run(self, wave, wfe_file, plot=True, verbose=True):
    """
      Run the simulation.
    """
    # do a few sanity checks first
    try:
      self.PUPIL_SAMPLING 	= int(self.PUPIL_SAMPLING)
      self.PUPIL_GAMMA 		= int(self.PUPIL_GAMMA)
    except ValueError:
      self.logger.critical(" PUPIL_SAMPLING and PUPIL_GAMMA should be an integer!")
      exit(0)
      
    try:
      assert self.PUPIL_GAMMA % 2 == 0
    except AssertionError:
      self.logger.warning(" Pupil gamma should be even. Could produce unexpected results.")    
      
    try:
      assert self.NSLICES % 2 == 1
    except AssertionError:
      self.logger.critical(" Number of slices should be odd!")
      exit(0)
      
    try:
      assert isPowerOfTwo(self.PUPIL_SAMPLING) == True
    except AssertionError:
      self.logger.critical(" Pupil sampling should be a power of two!")
      exit(0)
      
    # instantiate camera, entrance pupil and output
    cam = camera(self.CAMERA_FWNO)
    pupil = circular(self.logger, cam, self.PUPIL_SAMPLING, self.PUPIL_GAMMA, self.PUPIL_RADIUS, verbose=verbose)  
    this_composite_image = self.datacube.composite(self.datacube, wave, pupil)
      
    # get WFE if requested
    if self.ADD_WFE:
      wfe = zwfe(wfe_file, logger, verbose=verbose)
      wfe.parse()
      wfe_h = wfe.getHeader()	
      wfe_d = wfe.getData(in_radians=True, pad_pupil=pupil)		# returns data same dimensions as pupil array, in radians
	  
      if wfe_h['SAMPLING'][0] != self.PUPIL_SAMPLING:
	self.logger.critical(" Zemax WFE sampling is not the same as the pupil sampling! (" + str(wfe_h['SAMPLING'][0]) + " != " + str(self.PUPIL_SAMPLING) + ")")
	exit(0)
      pl.addImagePlot("wfe (radians)", np.abs(np.fft.fftshift(wfe_d)), extent=pupil.getExtent(), xl='mm', yl='mm')  

    # rescale image to same plate scale as resampled_im
    im = pupil.toConjugateImage(wave, verbose=True)
    d, hfov = im.getAmplitudeScaledByAiryDiameters(3, normalise=True)
    pl.addImagePlot("-> fft to image space", d, extent=(-hfov, hfov, -hfov, hfov), xl="arcsec", yl="arcsec")
    im.resample(self.resampling_im.pscale, self.resampling_im.getDetectorHFOV(), verbose=True)
    pupil = im.toConjugatePupil(verbose=True)
    
    # slicing
    if self.DO_SLICING:
      # take slices from the image space
      slices = []
      for s in range(self.NSLICES):	
	im = pupil.toConjugateImage(wave)				# SLICING SPACE CHANGE. move from pupil to image space. centered DC.
	offset = (s-((self.NSLICES-1)/2))*self.SLICE_WIDTH
	im.sliceUp(self.SLICE_WIDTH, offset=offset, gamma=self.resampling_im.pupil.gamma, slice_number=s+1, verbose=True)	# create a new pupil conjugate image instance for each slice
	pl.addScatterPlot(None, [(-(self.SLICE_WIDTH*im.resolution_element)/2)+(offset*im.resolution_element), 
				(-(self.SLICE_WIDTH*im.resolution_element)/2)+(offset*im.resolution_element)], [-hfov, hfov], xr=(-hfov, hfov), yr=(-hfov, hfov), overplot=True)
	pl.addScatterPlot(None, [((self.SLICE_WIDTH*im.resolution_element)/2)+(offset*im.resolution_element), 
				((self.SLICE_WIDTH*im.resolution_element)/2)+(offset*im.resolution_element)], [-hfov, hfov], xr=(-hfov, hfov), yr=(-hfov, hfov), overplot=True)
	pl.addTextToPlot(hfov-(hfov/4), ((self.SLICE_WIDTH*im.resolution_element)/2)-((offset+0.5)*im.resolution_element), str(im.slice_number), color='w', fontsize=10)
	slices.append(im)
	    
      for s in slices:	
	 # fft back to pupil plane 
	new_pupil = s.toConjugatePupil()				# SLICING SPACE CHANGE. move from image to pupil space. zeroed DC.
        pl.addImagePlot("-> take slice " + str(s.slice_number) + " -> ifft to pupil space", new_pupil.getAmplitude(shift=True, normalise=True), 
			extent=new_pupil.getExtent(), xl='mm', yl='mm')
    	
        # add phase WFE 
        if self.ADD_WFE:
	  new_pupil.addToPhase(wfe_d)
	  plt_title_prefix = "added phase error "
	  self.logger.debug(" Added phase error for slice " + str(s.slice_number) + ".")
	else:
	  plt_title_prefix = ""
    
        # fft back to image plane
        im = new_pupil.toConjugateImage(wave)				# SLICING SPACE CHANGE. move from pupil to image space. centered DC.
        
        d, hfov = im.getAmplitudeScaledByAiryDiameters(3, normalise=True)
        pl.addImagePlot(plt_title_prefix + "-> fft to image space", d, extent=(-hfov, hfov, -hfov, hfov), xl="arcsec", yl="arcsec")
        this_composite_image.add(im)
    else:    
      # add phase WFE 
      if self.ADD_WFE:
        pupil.addToPhase(wfe_d) 
        plt_title_prefix = "added phase error "
        self.logger.debug(" Added phase error.")
      else:
	  plt_title_prefix = ""
	  
      # move from pupil to image space	
      im = pupil.toConjugateImage(wave)  				# NON-SLICING SPACE CHANGE. move from pupil to image space. centered DC.
      
      d, hfov = im.getAmplitudeScaledByAiryDiameters(3, normalise=True)
      pl.addImagePlot(plt_title_prefix + "-> fft to image space", d, extent=(-hfov, hfov, -hfov, hfov), xl="arcsec", yl="arcsec")
      
      this_composite_image.add(im)					# take amplitude and add per slice
    
    if plot:
      pl.draw(5,5)

    return this_composite_image

if __name__== "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-ws", help="wavelength start", default=800e-9, type=float)
  parser.add_argument("-we", help="wavelength end", default=801e-9, type=float)
  parser.add_argument("-wi", help="wavelength interval", default=25e-9, type=float)
  parser.add_argument("-e", help="zemax wfe file directory", default="etc/wfe/diffg", type=str)	# should be in format 
  parser.add_argument("-p", help="plot?", action="store_true")
  parser.add_argument("-f", help="create fits?", action="store_true")
  parser.add_argument("-fn", help="filename", action="store", default="cube.fits")
  parser.add_argument("-fv", help="view fits?", action="store_true")
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
  
  # setup plotting
  pl = plotter.plotter()	
  
  # get configuration parameters
  c = ConfigParser.ConfigParser()
  c.read("etc/default.ini")
  cfg = {}			# config parameters needed by __main__ *only*
  cfg_sim = {}			# config parameters needed by simulation instance
  cfg['RESAMPLING_FACTOR']	= int(c.get("output", "resampling_factor"))     
  cfg['HFOV']			= float(c.get("output", "hfov"))    
  
  cfg_sim['CAMERA_FWNO']	= float(c.get("camera", "wfno"))
  
  cfg_sim['PUPIL_SAMPLING']	= float(c.get("pupil", "pupil_sampling"))
  cfg_sim['PUPIL_GAMMA'] 	= float(c.get("pupil", "pupil_gamma"))
  cfg_sim['PUPIL_RADIUS'] 	= float(c.get("pupil", "pupil_radius_physical"))
  
  cfg_sim['ADD_WFE']		= bool(int(c.get("wfe", "do")))

  cfg_sim['DO_SLICING']		= bool(int(c.get("slicing", "do")))
  cfg_sim['NSLICES']		= int(c.get("slicing", "number"))
  cfg_sim['SLICE_WIDTH']	= float(c.get("slicing", "width"))				# in resolution elements
  cfg_sim['RESAMPLE_TO']	= float(c.get("slicing", "resample_to"))

  st = time.time()
  logger.debug(" Beginning simulation.")
  
  waves = np.arange(args.ws, args.we, args.wi)
  
  # first need to find the resampling parameters, we do this by create a conjugateImage instance
  logger.debug(" Ascertaining parameters to resample to " + sf(cfg_sim['RESAMPLE_TO']*10**9,3) + "nm")
  cam = camera(cfg_sim['CAMERA_FWNO'])
  resampling_pupil = circular(logger, cam, cfg_sim['PUPIL_SAMPLING'], cfg_sim['PUPIL_GAMMA'], cfg_sim['PUPIL_RADIUS'], verbose=True) 
  resampling_im = resampling_pupil.toConjugateImage(cfg_sim['RESAMPLE_TO'], verbose=True)
  
  s = sim(logger, plotter, resampling_im, len(waves), **cfg_sim)  
  for w in waves:
    logger.info(" !!! Processing for a wavelength of " + str(w*1e9) + "nm...")   
    
    if cfg_sim['ADD_WFE']:
      # find appropriate zemax wfe file
      wfe_file = None
      for f in os.listdir(args.e):
	f = args.e.rstrip('/') + '/' + f
	wfe = zwfe(f, logger, verbose=False)
	if wfe.parseFileHeader():
	  h = wfe.getHeader()
	  if np.isclose(h['WAVE']*h['WAVE_EXP'], w):
	    wfe_file = f
	    logging.debug(" Using WFE file " + wfe_file)
	    break
      if wfe_file is None:
	logger.critical(" Unable to find Zemax WFE file!")
	exit(0)
    else:
      wfe_file = None

    res = s.run(w, wfe_file, plot=args.p, verbose=args.v)
    s.datacube.addComposite(res)
    
  if args.f:
    s.datacube.write(args.fn, resampling_im, cfg['RESAMPLING_FACTOR'], cfg['HFOV'], verbose=True)
    if args.fv:
      d = pyds9.DS9()
      d.set("file composite.fits")
      d.set('cmap heat')
      d.set('scale log')
      d.set('zoom 4')
    
  fi = time.time()
  duration = fi-st
  logger.debug(" Full simulation completed in " + str(sf(duration, 4)) + "s.")

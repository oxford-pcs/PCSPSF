#!/usr/local/bin/python
# -*- coding: utf-8 -*-

'''
  simulate.py
  
  DESCRIPTION
  
  This program simulates the PSF resulting from slicing an image. Details are given in the README.md 
  for this module.

  NOTES
  
  EXAMPLES
  
  (using defaults)
  $ python sim.py -v
'''

import random
import logging
import copy
import os
import sys
import argparse
import time
from decimal import *

import pylab as plt
import numpy as np
import pyds9

import plotter
from pupil import circular
from camera import camera
from products import cube, composite_image
from util import sf, sort_zemax_wfe_files, read_psf_simulation_config_file, read_psf_simulation_parameters_file, is_power_of_two
from zmx_parser import zwfe
from editable import editable
from ui import ui

class sim():
  def __init__(self, logger, plotter, resampling_im, nwaves, pupil_physical_radius, cfg, p, args):
    self.logger 		= logger	
    self.plotter 		= plotter
    self.resampling_im		= resampling_im
    self.nwaves			= nwaves
    self.PUPIL_RADIUS		= pupil_physical_radius
    self.CAMERA_FWNO		= p['GENERAL']['CAMERA_WFNO']
    self.PUPIL_SAMPLING 	= cfg['PUPIL_SAMPLING']
    self.PUPIL_GAMMA		= cfg['PUPIL_GAMMA']
    self.ADD_CAM_WFE		= args.caw
    self.ADD_COL_WFE		= args.cow
    self.NSLICES		= p['GENERAL']['NSLICES']
    self.SLICE_WIDTH		= cfg['SLICE_WIDTH']
    self.RESAMPLE_TO		= cfg['RESAMPLE_TO']
    self.CAM_WFE_DATA		= p['CAM_WFE_DATA']
    self.COL_WFE_DATA		= p['COL_WFE_DATA']
    self.EPD			= p['GENERAL']['EPD']
    
    self.datacube		= cube(self.logger, [self.nwaves, self.PUPIL_SAMPLING*self.PUPIL_GAMMA, self.PUPIL_SAMPLING*self.PUPIL_GAMMA], self.resampling_im)
   
  def run(self, wave, verbose=True): 
    #  Sanity checks.
    #
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
      assert is_power_of_two(self.PUPIL_SAMPLING) == True
    except AssertionError:
      self.logger.critical(" Pupil sampling should be a power of two!")
      exit(0)
      
    #  Instantiate camera (to convert to spatial scale), entrance pupil and image instance for this wavelength.
    #
    cam = camera(self.CAMERA_FWNO)
    pupil = circular(self.logger, cam, self.PUPIL_SAMPLING, self.PUPIL_GAMMA, self.PUPIL_RADIUS, verbose=verbose)
    this_composite_image = composite_image(self.logger, [self.PUPIL_SAMPLING*self.PUPIL_GAMMA, self.PUPIL_SAMPLING*self.PUPIL_GAMMA], wave, pupil)
    self.plotter.addImagePlot("pupil", pupil.getAmplitude(shift=True, normalise=True), extent=pupil.getExtent(), xl='mm', yl='mm')
      
    '''
      Rescale pupil to yield same plate scale as [resampled_im]. 
      
      To do this, the corresponding angular hfov and plate scale is calculated for the current FFT'ed pupil (i.e. 
      in image space), and resampled to match the parameters from the prespecified [resampled_im].
    '''
    im = pupil.toConjugateImage(wave, verbose=True)
    d, hfov = im.getAmplitudeScaledByAiryDiameters(3, normalise=True)
    im.resample(self.resampling_im.pscale, self.resampling_im.getDetectorHFOV(), verbose=True)
    pupil = im.toConjugatePupil(verbose=True)
    self.plotter.addImagePlot("rescaled pupil", pupil.getAmplitude(shift=True, normalise=True), extent=pupil.getExtent(), xl='mm', yl='mm')
    self.plotter.addImagePlot("-> fft to image space", d, extent=(-hfov, hfov, -hfov, hfov), xl="arcsec", yl="arcsec")
    
    #  Slice the FoV up and add WFE maps independently for each field.
    #
    slices = []
    for s in range(self.NSLICES):	
      im = pupil.toConjugateImage(wave)					# SLICING SPACE CHANGE. move from pupil to image space. centered DC.
      offset = (s-((self.NSLICES-1)/2))*self.SLICE_WIDTH
      im.sliceUp(self.SLICE_WIDTH, offset=offset, gamma=self.resampling_im.pupil.gamma, slice_number=s, verbose=True)	# create a new pupil conjugate image instance for each slice
      self.plotter.addScatterPlot(None, [(-(self.SLICE_WIDTH*im.resolution_element)/2)+(offset*im.resolution_element), 
			      (-(self.SLICE_WIDTH*im.resolution_element)/2)+(offset*im.resolution_element)], [-hfov, hfov], xr=(-hfov, hfov), yr=(-hfov, hfov), overplot=True)
      self.plotter.addScatterPlot(None, [((self.SLICE_WIDTH*im.resolution_element)/2)+(offset*im.resolution_element), 
			      ((self.SLICE_WIDTH*im.resolution_element)/2)+(offset*im.resolution_element)], [-hfov, hfov], xr=(-hfov, hfov), yr=(-hfov, hfov), overplot=True)
      self.plotter.addTextToPlot(hfov-(hfov/4), ((self.SLICE_WIDTH*im.resolution_element)/2)-((offset+0.5)*im.resolution_element), str(im.slice_number), color='w', fontsize=10)
      slices.append(im)
	  
    for s in slices:	
      # FFT to pupil plane 
      #
      new_pupil = s.toConjugatePupil()					# SLICING SPACE CHANGE. move from image to pupil space. zeroed DC.
      self.plotter.addImagePlot("-> take slice " + str(s.slice_number) + " -> ifft to pupil space", new_pupil.getAmplitude(shift=True, normalise=True), 
		      extent=new_pupil.getExtent(), xl='mm', yl='mm')
       
      # Add phase error
      #
      # COLLIMATOR
      #
      if self.ADD_COL_WFE:
	this_wfe_file = None
	for wfe in self.COL_WFE_DATA:
	  if wave == Decimal(str(wfe['WAVE'])) and s.slice_number == wfe['SLICE_INDEX']:
	    this_wfe_file = wfe['PATH']
	    self.logger.debug(" Using collimator WFE map from file " + this_wfe_file)
	if this_wfe_file == None:
	  self.logger.critical(" Didn't find suitable WFE map!")
	  exit(0)	  
	wfe = zwfe(this_wfe_file, self.logger, verbose=verbose)
	wfe.parse()
	wfe_h = wfe.getHeader()	
	wfe_d = wfe.getData(self.EPD, match_pupil=pupil, in_radians=True)							# entrance pupil
	
	self.plotter.addImagePlot("wfe (radians)", np.abs(np.fft.fftshift(wfe_d)), extent=pupil.getExtent(), xl='mm', yl='mm')  

	new_pupil.convolve(wfe_d)
	plt_title_prefix = "added phase error "
	self.logger.debug(" Added collimator phase error for slice " + str(s.slice_number) + ".")
      else:
	plt_title_prefix = ""
        
      # CAMERA  
      #
      if self.ADD_CAM_WFE:
	this_wfe_file = None
	for wfe in self.CAM_WFE_DATA:
	  if wave == Decimal(str(wfe['WAVE'])) and s.slice_number == wfe['SLICE_INDEX']:
	    this_wfe_file = wfe['PATH']
	    self.logger.debug(" Using camera WFE map from file " + this_wfe_file)
	if this_wfe_file == None:
	  self.logger.critical(" Didn't find suitable WFE map!")
	  exit(0)	  
	wfe = zwfe(this_wfe_file, self.logger, verbose=verbose)
	wfe.parse()
	wfe_h = wfe.getHeader()	
	wfe_d = wfe.getData(self.EPD, match_pupil=pupil, in_radians=True)							# entrance pupil
	
	self.plotter.addImagePlot("wfe (radians)", np.abs(np.fft.fftshift(wfe_d)), extent=pupil.getExtent(), xl='mm', yl='mm')  

	new_pupil.convolve(wfe_d)
	plt_title_prefix = "added phase error "
	self.logger.debug(" Added camera phase error for slice " + str(s.slice_number) + ".")
      else:
	plt_title_prefix = ""
  
      # FFT to image plane
      #
      im = new_pupil.toConjugateImage(wave)				# SLICING SPACE CHANGE. move from pupil to image space. centered DC.
      
      d, hfov = im.getAmplitudeScaledByAiryDiameters(3, normalise=True)
      self.plotter.addImagePlot(plt_title_prefix + "-> fft to image space", d, extent=(-hfov, hfov, -hfov, hfov), xl="arcsec", yl="arcsec")
      this_composite_image.addSlice(im)
        
    return this_composite_image
  
def run(args, logger, plotter):
  # Read config file and simulation parameters.
  #
  cfg = read_psf_simulation_config_file(logger, args.c)
  p = read_psf_simulation_parameters_file(logger, args.s)
  xtra_header_keys = {}		# this is mostly used to record RMS phase error for each wavelength
  
  st = time.time()
  logger.debug(" Beginning simulation")
  
  # Get wavelength range and determine pupil radius from detector pixel size
  #
  waves = np.arange(Decimal(str(p['GENERAL']['WAVELENGTH_START'])), 
		    Decimal(str(p['GENERAL']['WAVELENGTH_END']+p['GENERAL']['WAVELENGTH_INTERVAL'])), 
		    Decimal(str(p['GENERAL']['WAVELENGTH_INTERVAL'])), dtype=Decimal)
  focal_ratio = (2*float(args.d))/cfg['PUPIL_REFERENCE_WAVELENGTH']
  pupil_physical_diameter = p['GENERAL']['CAMERA_EFFL']/focal_ratio
  pupil_physical_radius = pupil_physical_diameter/2
  xtra_header_keys['STAPUDIA'] = (pupil_physical_diameter, "starting pupil physical diameter (mm)")

  # Find parameters with which we will rescale the image.
  #
  '''
    First, as the angular size of the resolution element is dependent on lambda, we need to define a reference
    system through which we can resample each wavelength. The wavelength this is done for is defined in the 
    configuration file as [pupil.resample_to]. This is done by establishing the FoV of the FFT'ed pupil grid and 
    the corresponding angular plate scale ("/px) for both the reference and wavelength being considered and 
    interpolating accordingly. To avoid extrapolation, this reference wavelength should be at a wavelength that 
    is blueward of the smallest wavelength to be considered.
    
    All the information required to rescale is held in the [resampling_im] instance.
  '''
  logger.debug(" Ascertaining parameters to resample to " + str(cfg['RESAMPLE_TO']*Decimal('1e9')) + "nm")
  cam = camera(p['GENERAL']['CAMERA_WFNO'])
  resampling_pupil = circular(logger, cam, cfg['PUPIL_SAMPLING'], cfg['PUPIL_GAMMA'], pupil_physical_radius, verbose=True) 
  resampling_im = resampling_pupil.toConjugateImage(cfg['RESAMPLE_TO'], verbose=True)
  
  # Run simulation.
  # 
  # The result from the simulation is an image instance, which we then add to a datacube.
  #
  s = sim(logger, plotter, resampling_im, len(waves), pupil_physical_radius, cfg, p, args)  
  
  for idx, w in enumerate(waves):
    logger.info(" !!! Processing for a wavelength of " + str(float(w)*1e9) + "nm...") 
    
    this_st = time.time()
    composite_image = s.run(w, verbose=args.v)
    plotter.addImagePlot("composite image (may require zoom!)", composite_image.getAmplitude(normalise=True), extent=composite_image.getExtent(), xl='arcsec', yl='arcsec')
    if args.p:
      plotter.draw(5,5)				#FIXME: static cols/rows
    this_duration = time.time()-this_st
    
    if idx>0:
      logger.debug(" Last wavelength iteration took " + str(int(this_duration)) + "s, expected time to completion is " + str(int(((len(waves)-(idx+1))*this_duration))) + "s.")
    
    xtra_header_keys['RMSP' + str(idx)] = (composite_image.getRMSPhase(), 'RMS Phase Error for wavelength ' + str(idx) + ' (wv)')
    s.datacube.addImage(composite_image)	# this call adds the returned image instance to the datacube
  
  # Make and view output.
  #
  if args.f:
    s.datacube.resampleAndCrop(cfg['RESAMPLING_FACTOR'], cfg['HFOV'])
    s.datacube.write(args.fn, cfg, p, args, xtra_header_keys)
    if args.fv:
      d = pyds9.DS9()
      d.set("file " + args.fn)
      d.set('cmap heat')
      d.set('scale log')
      d.set('zoom 4')
    
  duration = time.time()-st
  logger.debug(" This simulation completed in " + str(sf(duration, 4)) + "s.")

if __name__== "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", help="simulation configuration file path (.ini)", default="etc/default.ini", type=str)
  parser.add_argument("-s", help="simulation parameters file path (.json)", default="/local/home/barnsley/metadata/1/config.json", type=str)
  parser.add_argument("-p", help="plot?", action="store_true")
  parser.add_argument("-f", help="create fits file?", action="store_true")
  parser.add_argument("-fn", help="filename", action="store", default="cube.fits")
  parser.add_argument("-fv", help="view cube?", action="store_true")
  parser.add_argument("-v", help="verbose", action="store_true")
  parser.add_argument("-d", help="detector pixel pitch", type=Decimal, default='15e-6')
  parser.add_argument("-caw", help="add WFE error", action="store_true")
  parser.add_argument("-cow", help="add WFE error", action="store_true")
  args = parser.parse_args()
  
  #  Setup logger and plotter.
  #
  logger = logging.getLogger(args.fn)
  logger.setLevel(logging.DEBUG)
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  formatter = logging.Formatter("(" + str(os.getpid()) + ") %(asctime)s:%(levelname)s: %(message)s")
  ch.setFormatter(formatter)
  logger.addHandler(ch)
  plotter = plotter.plotter()

  run(args, logger, plotter)

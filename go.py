#!/usr/local/bin/python
# -*- coding: utf-8 -*-
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
import pyzdde.zdde as pyz    

from simulation import sim
import plotter
from pupil import *
from camera import pcamera
from products import cube
from util import sf, readConfigFile, isPowerOfTwo
from zSpec.spectrograph import Spectrograph
from zSpec.spectrograph_config_manager.slit import slit
from zSpec.spectrograph_config_manager.detector import detector 
from zSpec.zController.Controller import Controller

def run(args, logger, plotter):
  
  # Read config file and simulation parameters and initial setup.
  #
  cfg = readConfigFile(logger, args.c)
  xtra_header_keys = {} # used to record RMS phase error for each wavelength
  st = time.time()
  
  logger.debug(" Beginning simulation")
  
  # Create link to Zemax and assemble spectrograph model.
  #
  zmx_link = pyz.createLink()
  zcontroller = Controller(zmx_link)
  
  s = Spectrograph(args.co, args.ca, zcontroller)
  
  slit_pattern = slit(args.sf, args.s)
  pattern_data = slit_pattern.cfg['pattern_data']
  detector_cfg = detector(args.df, args.d)
  detector_data = detector_cfg.cfg['detector_data']
  
  # Get wavelength range 
  #
  waves = np.arange(Decimal(args.ws), Decimal(args.we + args.wi), 
                    Decimal(args.wi), dtype=Decimal)
  
  # Determine the smallest pupil size (D_pupil) that corresponds to Nyquist 
  # sampling at the detector given a camera focal length, f_cam, and some 
  # reference wavelength, lambda. As we want to be AT LEAST Nyquist sampled at 
  # all wavelengths, we really want to ensure that the system is Nyquist 
  # sampling at the longest wavelength. Doing so means that shorter wavelengths 
  # will be oversampled. The wavelength at which the pupil is constructed for 
  # can be adjusted in the configuration file as [pupil.reference_wavelength].
  # 
  # We get D_pupil by equating the spatial size of 2 detector pixels (D_p) with
  # the spatial size of one resolution element, i.e.
  # 
  # 2 x D_p = lambda*f_cam / D_pupil
  # 
  # For a SWIFT-type spectrograph, each slice has its own "mini pupil" which is 
  # focussed by a microlens to produce the exit slit. The diameter of the this 
  # mini pupil is what is being considered here, and we consider this position 
  # to be the start of the simulation.
  #
  camera_EFFL = s.camera.getEFL(wavelength=cfg['PUPIL_REFERENCE_WAVELENGTH'])
  pupil_physical_diameter = (cfg['PUPIL_REFERENCE_WAVELENGTH']* camera_EFFL)/ \
    (2*float(detector_data['pitch']))
  pupil_physical_radius = (pupil_physical_diameter/2)
  
  xtra_header_keys['EPD'] = (pupil_physical_diameter*1e3, 
                                  "physical entrance pupil diameter (mm)")
  
  # Find parameters with which we will rescale the image.
  # 
  # Why?
  #
  # As the angular size of the resolution element is dependent on lambda, 
  # we need to define a reference system through which we can resample each 
  # wavelength. The wavelength this is done for is defined in the configuration
  # file as [pupil.resample_to_wavelength]. 
  #
  # To do this, we establish the FoV of the FFT'ed pupil grid and the 
  # corresponding angular plate scale ("/px) for the reference wavelength. The
  # latter is then used to determine the scale factor which needs to be applied 
  # to the wavelength currently being considered. To avoid extrapolation, this 
  # reference wavelength should be blueward of the smallest wavelength to be 
  # considered.
  #  
  # All the information required to rescale is held in the [resampling_im] 
  # instance.
  #
  logger.debug(" Ascertaining parameters to resample to " + 
               str(cfg['PUPIL_RESAMPLE_TO_WAVELENGTH']*Decimal('1e9')) + "nm")
  
  resampling_pupil = pupil_circular(logger, cfg['PUPIL_SAMPLING'], 
                                    cfg['PUPIL_GAMMA'], pupil_physical_radius, 
                                    verbose=True) 
  
  cam_EFL = s.camera.getEFL(wavelength=cfg['PUPIL_RESAMPLE_TO_WAVELENGTH'])
  cam = pcamera(cam_EFL, pupil_physical_radius*2)
  resampling_im = resampling_pupil.toConjugateImage(
    cfg['PUPIL_RESAMPLE_TO_WAVELENGTH'], 
    cam, verbose=True)
  
  # Init datacube and run simulations for each wavelength.
  # 
  # The result from a simulation is an image instance which we can append 
  # to the datacube.
  #
  dcube = cube(logger, dshape=resampling_im.data.shape)
  s = sim(logger, plotter, resampling_im, resampling_pupil, len(waves), 
          cam, cfg, args)  
  
  for idx, w in enumerate(waves):
    logger.info(" !!! Processing for a wavelength of " + str(float(w)*1e9) + 
                "nm...") 
    
    this_st = time.time()
    dcube.append(s.run(w, verbose=args.v))
    this_duration = time.time()-this_st
    
    if idx>0:
      logger.debug(" Last wavelength iteration took " + str(int(this_duration)) 
                   + "s, expected time to completion is " + 
                   str(int(((len(waves)-(idx+1))*this_duration))) + "s.")
    
  # Make and view output.
  #
  if args.f:
    dcube.write(args, cfg, xtra_header_keys)
    if args.fv:
      import pyds9
      d = pyds9.DS9()
      d.set("file " + args.fn)
      d.set('cmap heat')
      d.set('scale log')
      d.set('zoom 4')
    
  duration = time.time()-st
  logger.debug(" This simulation completed in " + str(sf(duration, 4)) + "s.")
  
  zmx_link.close()

if __name__== "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", help="simulation configuration file path (.ini)", default="etc/default.ini", type=str)
  parser.add_argument("-co", help="camera ZEMAX file", default="C:\\Users\\barnsley\\Google Drive\\Spectrograph_Optics\\collimator60.ZMX")   
  parser.add_argument("-ca", help="camera ZEMAX file", default="C:\\Users\\barnsley\\Google Drive\\Spectrograph_Optics\\camera70.ZMX")    
  parser.add_argument("-ws", help="wavelength start (m)", default="650e-9", type=Decimal)
  parser.add_argument("-we", help="wavelength end (m)", default="1000e-9", type=Decimal)
  parser.add_argument("-wi", help="wavelength interval (m)", default="175e-9", type=Decimal)
  parser.add_argument("-s", help="slit name", default="SWIFT")
  parser.add_argument("-d", help="detector name", default="apogee")
  parser.add_argument("-sf", help="slits file", default="zSpec\\spectrograph_config_manager\\slits.json")
  parser.add_argument("-df", help="detector file", default="zSpec\\spectrograph_config_manager\\detectors.json")  
  parser.add_argument("-p", help="plot?", action="store_true")
  parser.add_argument("-f", help="create fits file?", action="store_true")
  parser.add_argument("-fn", help="filename", action="store", default="cube.fits")
  parser.add_argument("-fv", help="view cube?", action="store_true")
  parser.add_argument("-v", help="verbose", action="store_true")
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

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

from simulation import sim
import plotter
from pupil import *
from reimager import reimager
from products import cube
from util import sf, readConfigFile, isPowerOfTwo
from zSpec import zSpectrograph
from instrument_builder import SWIFT_like

def run(args, logger, plotter):
  # Read config file for simulation parameters.
  #
  cfg = readConfigFile(logger, args.c)
  xtra_header_keys = {} # used to record RMS phase error for each wavelength
  st = time.time()
  
  logger.debug(" Beginning simulation")
  
  # Build Zemax spectrograph model
  #
  zspec = zSpectrograph(cfg['SIM_COLLIMATOR_ZMX_FILE'], 
    cfg['SIM_CAMERA_ZMX_FILE'])

  # Build the instrument.
  #
  # Although we construct the components from the configuration files, we 
  # override some attributes.
  #
  # -> Assign camera/collimator EFFL to match Zemax models.
  # -> Set the preoptics anamorphic magnification to 1
  # -> Set the preoptics WFNO such that a spatial resolution element spans 
  #    [slices_per_resel] slices.
  # 
  instrument = SWIFT_like(cfg['PREOPTICS_CFG_NAME'], 
    cfg['IFU_CFG_NAME'], 
    cfg['SPECTROGRAPH_CFG_NAME'], 
    cfg['DETECTOR_CFG_NAME'], 
    config_dir=cfg['SIM_INSTRUMENT_CONFIGS_DIR_PATH'],
    logger=logger)

  zspec_attr = zspec.getSystemAttr(cfg['PUPIL_REFERENCE_WAVELENGTH'])
  instrument.spectrograph.cfg['camera_EFFL'] = zspec_attr['camera_EFFL']
  instrument.spectrograph.cfg['collimator_EFFL'] = zspec_attr['collimator_EFFL']

  instrument.preoptics.cfg['magnification_across_slices'] = \
    instrument.preoptics.cfg['magnification_along_slices']

  instrument.preoptics.cfg['WFNO'] = cfg['IFU_SLICES_PER_RESEL'] * \
    instrument.ifu.cfg['slice_width_physical'] / \
    cfg['PUPIL_REFERENCE_WAVELENGTH']

  instrument.assemble()

  # Get wavelength range as decimal.
  #
  waves = np.arange(cfg['SIM_WAVELENGTH_START'],
    Decimal(cfg['SIM_WAVELENGTH_END'] + \
    cfg['SIM_WAVELENGTH_INTERVAL']),
    Decimal(cfg['SIM_WAVELENGTH_INTERVAL']), dtype=Decimal)

  # Determine the smallest pupil size (D_pupil) that corresponds to Nyquist 
  # sampling at the detector given a camera focal length, f_cam, and some 
  # reference wavelength, lambda. This overrides any pupil diameter set in 
  # the preoptics.
  #
  # As we want to be AT LEAST Nyquist sampling at all wavelengths, we really
  # want to ensure that the system is Nyquist sampling at the shortest 
  # wavelength. Doing so means that longer wavelengths will be oversampled. 
  # The wavelength at which the pupil is constructed for can be adjusted in 
  # the configuration file as [pupil.reference_wavelength].
  # 
  # We get [D_pupil], the pupil size for a single slice, by equating the spatial
  # size of 2 detector pixels (D_p) with the spatial size of one resolution 
  # element, i.e.
  # 
  #                  2 x D_p = lambda*f_cam / D_pupil
  # 
  pupil_physical_diameter = (cfg['PUPIL_REFERENCE_WAVELENGTH'] * \
    instrument.camera_EFFL)/(2*instrument.detector_pixel_pitch)
  pupil_physical_radius = (pupil_physical_diameter/2)
  
  xtra_header_keys['EPD'] = (pupil_physical_diameter*1e3,
    "physical entrance pupil diameter (mm)")
  
  # Now find parameters with which we will rescale the image.
  #
  # As the angular size (and thus spatial size) of the resolution element is 
  # dependent on lambda, we need to define a reference system through which we 
  # can resample each wavelength. The wavelength this is done for is defined 
  # in the configuration file as [pupil.resample_to_wavelength]. 
  #
  # To do this, we first stablish the spatial pixel scale (m/px) for the 
  # reference wavelength. The latter is then used to determine the scale factor 
  # which needs to be applied to the wavelength currently being considered. To
  # avoid extrapolation, this reference wavelength should be blueward of the 
  # smallest wavelength to be considered.
  #  
  # All the information required to rescale is held in the [resampling_im] 
  # instance.
  #
  logger.debug(" Ascertaining parameters to resample to " + \
    str(cfg['PUPIL_RESAMPLE_TO_WAVELENGTH']*Decimal('1e9')) + "nm")
  
  resampling_pupil = pupil_circular(logger, cfg['PUPIL_SAMPLING'],
    cfg['PUPIL_GAMMA'], pupil_physical_radius, verbose=True) 

  preoptics_reimager = reimager(instrument.preoptics_WFNO)

  resampling_im = resampling_pupil.toConjugateImage(
    cfg['PUPIL_RESAMPLE_TO_WAVELENGTH'], 
    preoptics_reimager, verbose=True)
  
  # Init datacube and run simulations for each wavelength.
  # 
  # The result from a simulation is an image instance which we can append 
  # to the datacube.
  #
  dcube = cube(logger, dshape=resampling_im.data.shape)
  s = sim(logger, plotter, resampling_im, resampling_pupil, len(waves), 
          preoptics_reimager, zspec, cfg, instrument)  
  
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
    if args.d:
      import pyds9
      d = pyds9.DS9()
      d.set("file " + args.o)
      d.set('cmap heat')
      d.set('scale log')
      d.set('zoom 4')
    
  duration = time.time()-st
  logger.debug(" This simulation completed in " + str(sf(duration, 4)) + "s.")

if __name__== "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", help="simulation configuration file path (.ini)", 
    default="etc/default.ini", type=str)
  parser.add_argument("-p", help="plot?", action="store_true")
  parser.add_argument("-f", help="create fits file?", action="store_true")
  parser.add_argument("-o", help="output filename", action="store", 
    default="cube.fits")
  parser.add_argument("-d", help="display fits datacube? (-f must be set)", 
    action="store_true")
  parser.add_argument("-v", help="verbose", action="store_true")
  args = parser.parse_args()
  
  #  Setup logger and plotter.
  #
  logger = logging.getLogger(args.o)
  logger.setLevel(logging.DEBUG)
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  formatter = logging.Formatter("(" + str(os.getpid()) + \
    ") %(asctime)s:%(levelname)s: %(message)s")
  ch.setFormatter(formatter)
  logger.addHandler(ch)
  plotter = plotter.plotter()

  run(args, logger, plotter)

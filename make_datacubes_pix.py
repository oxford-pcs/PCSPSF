#!/usr/local/bin/python
# -*- coding: utf-8 -*-

'''
  make_datacubes_pix.py
  
  DESCRIPTION
  
  This wrapper program calls simulate.py asynchronously for different pixel pitches, generating
  datacubes both with and without wfe.

  NOTES
  
  EXAMPLES
  
  (using defaults)  
  python gen_datacubes.py
  
'''

import argparse
from decimal import Decimal
from multiprocessing import Pool
import subprocess
import logging
import os

import numpy as np

from simulate import run

if __name__== "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", help="simulation configuration file path (.ini)", default="etc/default.ini", type=str)
  parser.add_argument("-s", help="simulation parameters file path (.json)", default="/home/barnsley/ELT-PCS/scripts/metadata/2/config.json", type=str)
  parser.add_argument("-v", help="verbose", action="store_true")
  parser.add_argument("-p", help="plot?", action="store_true")
  parser.add_argument("-fv", help="view cubes?", action="store_true")
  parser.add_argument("-fp", help="file prefix", default="cube_", type=str)
  parser.add_argument("-o", help="output directory", default="out", type=str)
  parser.add_argument("-ds", help="detector pixel pitch start", default="9e-6", type=Decimal)
  parser.add_argument("-de", help="detector pixel pitch end", default="15e-6", type=Decimal)
  parser.add_argument("-di", help="detector pixel pitch interval", default="3e-6", type=Decimal)
  parser.add_argument("-nc", help="number of processes (or cores)", default=4, type=int)
  args = parser.parse_args()
  
  #  Setup logger.
  #
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  formatter = logging.Formatter("(" + str(os.getpid()) + ") %(asctime)s:%(levelname)s: %(message)s")
  ch.setFormatter(formatter)
  logger.addHandler(ch)
  
  # Make output directory.
  os.mkdir(args.o)
  
  # Set some mandatory arguments
  #
  args.f = True		# we always want to create a fits file	
  
  # Set up range of detector pixel pitches to process
  #
  pitches = np.arange(args.ds, args.de+args.di, args.di, dtype=Decimal)
  
  # Farm off subprocesses to worker pool
  #
  workers = Pool(processes=args.nc)

  for p in pitches:
    # Set per pitch args parameters
    #
    args.d = p
    call = ["python", "simulate.py", 
            "-c", args.c, 
	    "-s", args.s,
	    "-d", str(args.d)]
    if args.fv:
      call.append("-fv")
    if args.p:
      call.append("-p")
    if args.f:
      call.append("-f")
    if args.v:
      call.append("-v")
    
    # NO WFE
    args.fn = args.o.rstrip('/') + "/" + args.fp + "nowfe_" + str(p*Decimal('1e6')) + ".fits"
    res = workers.apply_async(subprocess.call, [call + ['-fn', args.fn]]) # no worker callback called when complete  
    
    # COLLIMATOR WFE
    args.fn = args.o.rstrip('/') + "/" + args.fp + "wfe_col_" + str(p*Decimal('1e6')) + ".fits"
    res = workers.apply_async(subprocess.call, [call + ['-fn', args.fn, '-cow']]) # no worker callback called when complete  
    
    # CAMERA WFE
    args.fn = args.o.rstrip('/') + "/" + args.fp + "wfe_cam_" + str(p*Decimal('1e6')) + ".fits"
    res = workers.apply_async(subprocess.call, [call + ['-fn', args.fn, '-caw']]) # no worker callback called when complete  
    
    # BOTH WFE
    args.fn = args.o.rstrip('/') + "/" + args.fp + "wfe_colcam_" + str(p*Decimal('1e6')) + ".fits"
    res = workers.apply_async(subprocess.call, [call + ['-fn', args.fn, '-caw']]) # no worker callback called when complete  
   
  # Rejoin this main thread upon completion
  #
  workers.close()
  workers.join()  
    
  exit(0)
  


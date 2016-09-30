'''
  setup.py
  
  DESCRIPTION

  This utility program generates a JSON formatted configuration file required by the simulation.
  
  Only if a homogenous (in that they have the same field locations) set of WFE maps is found for each 
  of the wavelengths specified will the program be able to proceed. 
  
  NOTES
  
  The [WFE_DATA.SLICE_IDX] field will be added w/ the first slice index representing the top of the 
  field.
  
  TO DO
  
  EXAMPLES
  
  (using defaults)
  $ python setup.py -w
'''

import argparse
import logging
import json
from decimal import *
from collections import Counter
import os

import numpy as np

from util import sort_zemax_wfe_files, read_zemax_simulation_parameters_file
from zmx_parser import zsystemdata

if __name__== "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-ws", help="wavelength start", default="650e-9", type=Decimal)
  parser.add_argument("-we", help="wavelength end", default="1000e-9", type=Decimal)
  parser.add_argument("-wi", help="wavelength interval", default="25e-9", type=Decimal)
  parser.add_argument("-z", help="Zemax parameters file path", default="/local/home/barnsley/metadata/1/ZSIM_PARAMS.TXT", type=str)
  parser.add_argument("-f", help="make output file", action="store_true")
  parser.add_argument("-fn", help="output filename", action="store", default="config.json")
  args = parser.parse_args()
  
  # We assume that the other files are in the same parent directory as args.z
  #
  datadir = '/'.join(args.z.split('/')[:-1]) + '/'
  
  # Setup logger.
  # 
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  formatter = logging.Formatter("(" + str(os.getpid()) + ") %(asctime)s:%(levelname)s: %(message)s")
  ch.setFormatter(formatter)
  logger.addHandler(ch)

  waves = np.arange(args.ws, args.we+args.wi, args.wi, dtype=Decimal)
  
  # Read parameters used in Zemax.
  #
  params = read_zemax_simulation_parameters_file(logger, args.z)
  
  # Check wavelength requested ranges are within those defined in Zemax parameter file
  #
  if not all([waves[0] >= params['WAVE_START']*Decimal('1e-6'), waves[-1] <= params['WAVE_END']*Decimal('1e-6')]):
    logger.critical(' Requested wavelength range not within data\'s inherent wavelength range!')
    exit(0)
  
  # Search for WFE maps.
  #
  maps_col = sort_zemax_wfe_files(logger, datadir, params['WFE_COL_FILE_PREFIX'], waves, params['NSLICES'])
  maps_cam = sort_zemax_wfe_files(logger, datadir, params['WFE_CAM_FILE_PREFIX'], waves, params['NSLICES']) 
  
  if len(maps_col) != len(maps_cam):
    logger.critical(' Need same number of WFE maps for collimator and camera!')
    exit(0)
  
  # Generate output.
  #
  out_arr = [
    {"GENERAL": 
       {"SEARCH_DIRECTORY":datadir, 
	"NSLICES": params['NSLICES'], 
	"WAVELENGTH_START":float(args.ws), 
	"WAVELENGTH_END":float(args.we), 
	"WAVELENGTH_INTERVAL":float(args.wi),
	"SLICE_LENGTH": params['SLICE_LENGTH'],
	"INTER_SLICE_LENGTH": params['INTER_SLICE_LENGTH'],
	"SLICE_STAGGER": params['SLICE_STAGGER'],
	"WFE_SAMPLING": params['WFE_SAMPLING'],
	"COLLIMATOR_LENS_PATH": params['COLLIMATOR_LENS_PATH'],
	"CAMERA_LENS_PATH": params['CAMERA_LENS_PATH'],
	"CON_COLLIMATOR": params['CON_COLLIMATOR'],
	"CON_CAMERA": params['CON_CAMERA'],
	"CAMERA_WFNO": params['CAMERA_WFNO'],
	"CAMERA_EFFL": params['CAMERA_EFFL'],
	"EPD": params['EPD']
       }
    }, 
    {"COL_WFE_DATA": maps_col},
    {"CAM_WFE_DATA": maps_cam}
  ]
  print json.dumps(out_arr, indent=2)
    
  if args.f:
    with open(datadir + args.fn, 'w') as fp:
      json.dump(out_arr, fp, indent=2)

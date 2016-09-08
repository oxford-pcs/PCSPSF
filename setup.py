'''
  setup.py
  
  DESCRIPTION

  This utility program generates a JSON formatted configuration file required by the simulation.
  
  The configuration file contains information on:
  - The wavelengths to run the simulation over		-> GENERAL
  - The number of slices to use, if any			-> GENERAL
  - Slit details (width, separation, stagger)		-> GENERAL
  - Which Zemax configuration was used			-> GENERAL
  - Wavelength ranges					-> GENERAL
  - The entrance pupil diameter (from Zemax) 		-> ZSYSTEM_DATA
  - The WFNO of the system (from Zemax) 		-> ZSYSTEM_DATA
  - WFE maps (from Zemax) if flag is set		-> WFE_DATA

  Only if a homogenous (in that they have the same field locations) set of WFE maps is found for each 
  of the wavelengths specified will the program be able to proceed. 
  
  NOTES
  
  The [WFE_DATA.SLICE_IDX] field will be added with the negative-most field (depending on the sort flag) 
  indexed at 0. 
  
  TO DO
  
  EXAMPLES
  
  (using defaults)
  $ python setup.py -w -f 
'''

import argparse
import logging
import json
from decimal import *
from collections import Counter

import numpy as np

from util import sort_zemax_wfe_files, read_zemax_simulation_parameters_file
from zmx_parser import zsystemdata

if __name__== "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-ws", help="wavelength start", default="650e-9", type=Decimal)
  parser.add_argument("-we", help="wavelength end", default="675e-9", type=Decimal)
  parser.add_argument("-wi", help="wavelength interval", default="25e-9", type=Decimal)
  parser.add_argument("-z", help="Zemax parameters file path", default="/home/barnsley/ELT-PCS/scripts/metadata/2/ZSIM_PARAMS.TXT", type=str)
  parser.add_argument("-s", help="sort fields by x or y object angle", default='y', type=str)
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
  formatter = logging.Formatter("%(levelname)s:%(asctime)s:%(message)s")
  ch.setFormatter(formatter)
  logger.addHandler(ch)

  # Set up precision by which to interpret wavelengths.
  #
  waves = np.arange(args.ws, args.we+args.wi, args.wi, dtype=Decimal)
  
  # Read parameters used in Zemax.
  #
  params = read_zemax_simulation_parameters_file(logger, args.z)
  
  # Search for WFE maps.
  #
  res = sort_zemax_wfe_files(logger, datadir, params['WFE_FILE_PREFIX'], waves, params['NSLICES'])
  
  ## Establish which object angles to keep and filter result accordingly.
  ##
  counter_waves = Counter([r['WAVE'] for r in res])	# {WAVELENGTH1:n1, WAVELENGTH2:n2...}
  counter_fields = Counter([r['FIELD'] for r in res])
  
  fields_unique = []
  for f in counter_fields:
    fields_unique.append(f)
  if args.s == 'x':
    sort_idx = 0
  elif args.s == 'y':
    sort_idx = 1
  else:
    logger.critical(" Field sorting criteria not recognised!")  
    exit(0)
  y_object_angles_to_keep = sorted([f[sort_idx] for f in fields_unique])[((len(fields_unique)-1)/2)-((params['NSLICES']-1)/2):((len(fields_unique)-1)/2)+((params['NSLICES']-1)/2)+1]
  res = [r for r in res if r['FIELD'][sort_idx] in y_object_angles_to_keep]
  
  ## Check to see that we have the correct number of WFE maps.
  ## 
  if len(res) != len(waves)*params['NSLICES']:
    logger.critical(" Incorrect number of WFE maps found!")  
    exit(0) 
    
  logger.debug(" WFE maps sorted, assigning fields to following slices...") 
  
  ## Annotate fields with corresponding slice number. See NOTES.
  ##
  field_to_slice_translation = {}
  for s_idx, i in enumerate(sorted(set([r['FIELD'] for r in res]), key=lambda tup: tup[sort_idx])):
    field_to_slice_translation[i] = s_idx
    logger.debug(" Slice index " + str(s_idx) + " -> " + str(i))

  for r_idx, r in enumerate(res):
    res[r_idx]['SLICE_IDX'] = field_to_slice_translation[r['FIELD']] 

  # Generate output.
  #
  out_arr = [
    {"GENERAL": 
       {"SEARCH_DIRECTORY":datadir, 
	"SORT_FLAG":args.s, 
	"NSLICES": params['NSLICES'], 
	"WAVELENGTH_START":float(args.ws), 
	"WAVELENGTH_END":float(args.we), 
	"WAVELENGTH_INTERVAL":float(args.wi),
	"SLICE_WIDTH": params['SLICE_WIDTH'],
	"INTER_SLICE_WIDTH": params['INTER_SLICE_WIDTH'],
	"SLICE_STAGGER_WIDTH": params['SLICE_STAGGER_WIDTH'],
	"CON_COLLIMATOR": params['CON_COLLIMATOR'],
	"CON_CAMERA": params['CON_CAMERA'],
	"CAMERA_WFNO": params['CAMERA_WFNO'],
	"CAMERA_EFFL": params['CAMERA_EFFL'],
	"EPD": params['EPD']
       }
    }, 
    {"WFE_DATA": res}
  ]
  print json.dumps(out_arr, indent=2)
    
  if args.f:
    with open(datadir + args.fn, 'w') as fp:
      json.dump(out_arr, fp, indent=2)

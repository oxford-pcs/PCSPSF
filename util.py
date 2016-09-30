import codecs
import os
from collections import Counter
import ConfigParser
from decimal import *
import json

import numpy as np
import pylab as plt
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import median_filter, gaussian_filter

def _decode(encoding, fp):
  fp = codecs.open(fp, "r", encoding)
  content = fp.readlines()
  fp.close()
  return content

def is_power_of_two(num):
  while num % 2 == 0 and num > 1:
    num = num/2
  return num == 1

def read_psf_simulation_config_file(logger, path):
  '''
    Parses a config file into various dictionaries.
  '''
  c = ConfigParser.ConfigParser()
  c.read(path)
  
  cfg = {} 
  cfg['RESAMPLING_FACTOR']		= int(c.get("output", "resampling_factor"))     
  cfg['HFOV']				= float(c.get("output", "hfov"))    
  
  cfg['PUPIL_SAMPLING']			= int(c.get("pupil", "pupil_sampling"))
  cfg['PUPIL_GAMMA'] 			= int(c.get("pupil", "pupil_gamma"))
  cfg['PUPIL_REFERENCE_WAVELENGTH']	= float(c.get("pupil", "pupil_reference_wavelength"))
  cfg['RESAMPLE_TO']			= Decimal(c.get("pupil", "resample_to"))
 
  cfg['SLICE_WIDTH']			= float(c.get("slicing", "width"))
  
  return cfg

def read_psf_simulation_parameters_file(logger, path):
  '''
    Parses simulation parameters to various dictionaries.
  '''
  with open(path, 'r') as fp:
    p = json.load(fp)
  res = {}
  res['GENERAL'] 	= p[0]['GENERAL']
  res['COL_WFE_DATA']	= p[1]['COL_WFE_DATA']
  res['CAM_WFE_DATA']	= p[2]['CAM_WFE_DATA']
  return res

def read_zemax_simulation_parameters_file(logger, path):
  content = _decode("UTF-16-LE", path)
  res = {}
  for line in content:
    key = line.split()[0].strip(': ')
    val = line.split()[1].strip()
    
    if "NSLITLETS" in key:
      res['NSLICES'] = int(float(val))
    if "SLIT_LENGTH" in key:
      res['SLICE_LENGTH'] = float(val)
    if "INTER_SLIT_LENGTH" in key:
      res['INTER_SLICE_LENGTH'] = float(val)
    if "SLIT_STAGGER" in key:
      res['SLICE_STAGGER'] = float(val)
    if "CON_COLLIMATOR" in key:
      res['CON_COLLIMATOR'] = int(float(val))
    if "CON_CAMERA" in key:
      res['CON_CAMERA'] = int(float(val))
    if "WFE_SAMPLING" in key:
      res['WFE_SAMPLING'] = int(float(val))
    if "COLLIMATOR_LENS_PATH" in key:
      res['COLLIMATOR_LENS_PATH'] = str(val)
    if "CAMERA_LENS_PATH" in key:
      res['CAMERA_LENS_PATH'] = str(val)
    if "WFE_COL_FILE_PREFIX" in key:
      res['WFE_COL_FILE_PREFIX'] = str(val)
    if "WFE_CAM_FILE_PREFIX" in key:
      res['WFE_CAM_FILE_PREFIX'] = str(val)
    if "SYSTEM_DATA_FILE" in key:
      res['SYSTEM_DATA_FILE'] = str(val)   
    if "PARAMETERS_FILE" in key:
      res['PARAMETERS_FILE'] = str(val)   
    if "CAMERA_WFNO" in key:
      res['CAMERA_WFNO'] = float(val)
    if "CAMERA_EFFL" in key:
      res['CAMERA_EFFL'] = float(val)
    if "EPD" in key:
      res['EPD'] = float(val)
    if "WAVE_START" in key:
      res['WAVE_START'] = Decimal(str(val))   
    if "WAVE_END" in key:
      res['WAVE_END'] = Decimal(str(val))
    if "WAVE_INTERVAL" in key:
      res['WAVE_INTERVAL'] = Decimal(str(val))
      
  return res

def resample2d(i_data, i_s, i_e, i_i, o_s, o_e, o_i, kx=3, ky=3, s=0, gauss_sig=0, median_boxcar_size=0, clip=True):
  '''
    Resample a square 2D input grid with extents defined by [i_s] and [i_e] with 
    increment [i_i] to a new 2D grid with extents defined by [o_s] and [o_e] with 
    increment [o_i].
    
    returns: a 2D resampled array.
  '''
  
  # calculate bivariate spline, G,  using input grid and data
  grid_pre_rebin = np.arange(i_s, i_e, i_i)
  G = RectBivariateSpline(grid_pre_rebin, grid_pre_rebin, i_data, kx=kx, ky=ky)

  # evaluate this spline at new points on output grid
  grid_x, grid_y = np.mgrid[o_s:o_e:o_i, o_s:o_e:o_i]
  
  data = G.ev(grid_x, grid_y)
  
  if gauss_sig != 0:
    data = gaussian_filter(data, gauss_sig)
    
  if median_boxcar_size != 0:
    data = median_filter(data, median_boxcar_size)
    
  if clip:
    input_max = np.max(i_data)
    input_min = np.min(i_data)
    
    data[np.where(data>input_max)] = input_max
    data[np.where(data<input_min)] = input_min

  return data

def sf(fig, n):
  '''
    Truncates a number [fig] to [n] significant figures
    
    returns: string.
  '''
  format = '%.' + str(n) + 'g'
  return '%s' % float(format % float(fig))

def sort_zemax_wfe_files(logger, wfe_dir, prefix, waves, nfields):
  '''
    Search through a directory to find valid Zemax WFE files.
  '''
  from zmx_parser import zwfe				# need this here to avoid circular import
  logger.debug(" Searching directory " + wfe_dir + " for WFE maps...")
  res = []
  for f in os.listdir(wfe_dir):
    if f.endswith('~'):
      continue
    if not f.startswith(prefix):
      continue
    f_fullpath =  wfe_dir.rstrip('/') + '/' + f
    wfe = zwfe(f_fullpath, logger, verbose=False)
    logger.debug(" Attempting to parse file " + f)
    if wfe.parseFileHeader():
      logger.debug(" - This file has a valid WFE header")
      h = wfe.getHeader()
      logger.debug(" - Wavelength: " + str(sf(h['WAVE']*h['WAVE_EXP']*10**9, 3)) + "nm")
      found_wavelength_match = False			# establish if the wavelength from this WFE map corresponds to one requested in the simulation
      for w in waves:
        if h['WAVE']*h['WAVE_EXP'] == w:
	  found_wavelength_match = True
	  break
      if found_wavelength_match is False:
	logger.debug(" - Wavelength not found in requested list, ignoring")
	continue
      s = int(f.lstrip(prefix).split('_')[0])			# slice number from file name (PREFIX_SLICENUM_WAVELENGTH)
      logger.debug(" - Corresponds to requested wavelength of " + str(sf(w*10**9, 3)) + "nm")
      logger.debug(" - Field: " + str(h['FIELD'][0]) + ", " + str(h['FIELD'][1]))
      logger.debug(" - Slice Index: " + str(s))
      res.append({'PATH': f_fullpath, 'WAVE': float(w), 'FIELD': h['FIELD'], 'SLICE_INDEX': s})
    else:
      logger.debug(" - This is not a valid WFE map, ignoring")
  
  ## Check to see that we have the correct number of WFE maps.			
  ## TODO: really need more rigorous checking that we have the same number of wavelengths for each field etc.
  ## 
  if len(res) != len(waves)*nfields or len(res) != len(waves)*nfields:
    logger.critical(" Incorrect number of WFE maps found!")  
    exit(0) 

  return res


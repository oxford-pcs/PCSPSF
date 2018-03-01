import codecs
import os
from collections import Counter
import ConfigParser
from decimal import *
import json
from distutils.util import strtobool

import numpy as np
import pylab as plt
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import median_filter, gaussian_filter

def _decode(encoding, fp):
  fp = codecs.open(fp, "r", encoding)
  content = fp.readlines()
  fp.close()
  return content

def isPowerOfTwo(num):
  while num % 2 == 0 and num > 1:
    num = num/2
  return num == 1

def readConfigFile(logger, path):
  '''
    Parses a config file into various dictionaries.
  '''
  c = ConfigParser.ConfigParser()
  c.read(path)
  
  cfg = {} 

  cfg['SIM_CAMERA_ZMX_FILE']          = str(c.get("simulation", "camera_zmx_file"))
  cfg['SIM_COLLIMATOR_ZMX_FILE']      = str(c.get("simulation", "collimator_zmx_file"))
  cfg['SIM_WAVELENGTH_START']         = Decimal(c.get("simulation", "wavelength_start"))
  cfg['SIM_WAVELENGTH_END']           = Decimal(c.get("simulation", "wavelength_end")) 
  cfg['SIM_WAVELENGTH_INTERVAL']      = Decimal(c.get("simulation", "wavelength_interval")) 
  cfg['SIM_ADD_CAMERA_WFE']           = bool(strtobool(c.get("simulation", "add_camera_WFE"))) 
  cfg['SIM_ADD_COLLIMATOR_WFE']       = bool(strtobool(c.get("simulation", "add_collimator_WFE"))) 
  cfg['SIM_SLITS_FILE']               = str(c.get("simulation", "slits_file"))
  cfg['SIM_SLIT_NAME']                = str(c.get("simulation", "slit_name")) 
  cfg['SIM_DETECTORS_FILE']           = str(c.get("simulation", "detectors_file")) 
  cfg['SIM_DETECTOR_NAME']            = str(c.get("simulation", "detector_name")) 

  cfg['PUPIL_SAMPLING']               = int(c.get("pupil", "sampling"))
  cfg['PUPIL_WFE_MAP_SAMPLING']       = int(c.get("pupil", "zemax_WFE_map_sampling"))
  cfg['PUPIL_GAMMA']                  = int(c.get("pupil", "gamma"))
  cfg['PUPIL_REFERENCE_WAVELENGTH']   = float(c.get("pupil", 
                                                    "reference_wavelength"))
  cfg['PUPIL_RESAMPLE_TO_WAVELENGTH'] = Decimal(c.get("pupil", 
                                                      "resample_to_wavelength"))
 
  cfg['SLICE_RESEL_PER_SLICE']        = float(c.get("slicer", "resel_per_slice"))
  
  cfg['DET_PIXEL_PITCH']              = float(c.get("detector", "pixel_pitch"))

  cfg['OUTPUT_RESAMPLING_FACTOR']     = int(c.get("output", 
                                                  "resampling_factor"))     
  cfg['OUTPUT_HFOV']                  = float(c.get("output", "hfov"))    

  return cfg

def resample2d(i_data, i_s, i_e, i_i, o_s, o_e, o_i, kx=3, ky=3, s=0, 
               gauss_sig=0, median_boxcar_size=0, clip=True):
  '''
    Resample a square 2D input grid with extents defined by [i_s] and [i_e] with 
    increment [i_i] to a new 2D grid with extents defined by [o_s] and [o_e] 
    with increment [o_i].
    
    Returns a 2D resampled array, with options for smoothing (gaussian and 
    median) and clipping.
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

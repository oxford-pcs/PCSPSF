#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import random
import logging
import copy
import ConfigParser

import numpy as np

import plotter
from pupil import circle
from zmx_parser import zwfe
from util import sf

class sim():
  def __init__(self, logger, plotter, cfg_file):
    self.logger 	= logger
    self.plotter 	= plotter
    self.cfg_file	= cfg_file

  def run(self, wave, wfe_file, verbose=True):
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
    N_AIRY_DIAMETERS		= int(cfg.get("plotting", "n_airy_diameters"))

    try:
      PUPIL_SAMPLING 	= int(PUPIL_SAMPLING)
      PUPIL_GAMMA 	= int(PUPIL_GAMMA)
      assert PUPIL_GAMMA % 2 == 0
    except ValueError:
      self.logger.warning(" PUPIL_SAMPLING and PUPIL_GAMMA should be an integer.")
    except AssertionError:
      self.logger.warning(" Pupil gamma should be even.")    
    
    # construct entrance pupil
    pupil = circle(self.logger, PUPIL_SAMPLING, PUPIL_GAMMA, PUPIL_RADIUS, PUPIL_RADIUS_UNIT, wave, verbose=verbose)
    pl._addImagePlot("pupil", pupil.getData(), extent=pupil.getExtent(), xl=PUPIL_RADIUS_UNIT, yl=PUPIL_RADIUS_UNIT)

    # move from pupil to image space
    pupil.do2DFFT(shift=True)
    im_npix, detector_HFOV_scaled = pupil.getScaledPupilDescriptors(2)
    pl._addImagePlot("-> fft (amplitude)", pupil.getFFTAmplitude()[(pupil.getData().shape[0]/2)-(im_npix/2):(pupil.getData().shape[0]/2)+(im_npix/2), \
				  (pupil.getData().shape[0]/2)-(im_npix/2):(pupil.getData().shape[0]/2)+(im_npix/2)], \
                                  extent=(-detector_HFOV_scaled,detector_HFOV_scaled,-detector_HFOV_scaled,detector_HFOV_scaled), xl="arcsec", yl="arcsec")
    
    # take a slice corresponding to half a resolution element
    sw = pupil.takeSlice(.5)
    pl._addScatterPlot("", [-pupil.im_resolution_element/(2*sw),-pupil.im_resolution_element/(2*sw)], [-detector_HFOV_scaled,detector_HFOV_scaled], 
		       xr=(-detector_HFOV_scaled,detector_HFOV_scaled), yr=(-detector_HFOV_scaled,detector_HFOV_scaled), overplot=True)
    pl._addScatterPlot("", [pupil.im_resolution_element/(2*sw),pupil.im_resolution_element/(2*sw)], [-detector_HFOV_scaled,detector_HFOV_scaled], overplot=True)
        
    # fft back to pupil plane
    pupil.do2DFFT()
    pl._addImagePlot("-> slice -> fft", pupil.getFFTAmplitude(), extent=pupil.getExtent(), xl=pupil.rad_unit, yl=pupil.rad_unit)
    
    # add WFE
    wfe = zwfe(wfe_file, logger)
    wfe.parse()
    wfe_h = wfe.getHeader()
    wfe_d = wfe.getData()
    
    ##TODO: pad/trim if uspcaling or downscaling pupil!
    s = (pupil.rad*2)/wfe_h['EXIT_PUPIL_DIAMETER'] #how much bigger the pupil needs (multiple of pixel)
    
    p_g = (s)*PUPIL_SAMPLING		#TODO: mixture of sampling grids (pupil/wfe!)
    nzeros = int(round(p_g/2))
    
    df = np.fft.fft2(wfe_d)
    df = np.fft.fftshift(df)  
    
    #FIXME: doesn't preserve phase magnitude? (nzeros=0)
    #df2 = df[(df.shape[0]/2)/2:-(df.shape[0]/2)/2,(df.shape[0]/2)/2:-(df.shape[0]/2)/2]
    df2 = np.pad(df, nzeros, mode='constant')
    df2ifft = np.fft.ifft2(df2)
    df2ifft = np.abs(df2ifft)
    
    df2ifft = np.pad(df2ifft, 1024-((df2ifft.shape[0])/2), mode='constant')
    wfed_p = df2ifft*np.pi
    pl._addImagePlot("wavefront phase error", wfed_p, extent=(-pupil.physical_gsize/2,pupil.physical_gsize/2,-pupil.physical_gsize/2,pupil.physical_gsize/2),
		     xl=pupil.rad_unit, yl=pupil.rad_unit)
   
   
    pupil.addToPhase(wfed_p)
    
    # fft back to image plane
    pupil.do2DFFT()
    pl._addImagePlot("-> slice -> fft", pupil.getFFTAmplitude())
    
    pl.draw(3,3)

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
  s.run(801e-9, "etc/801_0_0_256x256")

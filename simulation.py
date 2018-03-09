import numpy as np
import pylab as plt

from products import cube
from util import isPowerOfTwo

class sim():
  def __init__(self, logger, plotter, resampling_im, resampling_pupil, nwaves, 
               preoptics_reimager, zspec, cfg, instrument):
    self.logger = logger
    self.plotter = plotter
    self.resampling_im = resampling_im
    self.resampling_pupil = resampling_pupil
    self.nwaves = nwaves
    self.preoptics_reimager = preoptics_reimager
    self.zspec = zspec
    self.cfg = cfg
    self.instrument = instrument

  def run(self, wave, verbose=True): 
    self.physical_pupil_diameter = self.resampling_pupil.physical_pupil_diameter
    self.pupil_sampling = self.resampling_pupil.sampling
    self.pupil_gamma = self.resampling_pupil.gamma
    
    #  Do some sanity checks.
    #
    try:
      int(self.resampling_pupil.sampling)
      int(self.resampling_pupil.gamma)
    except ValueError:
      self.logger.critical(" Pupil sampling and gamma should be integers!")
      exit(0)
      
    try:
      assert self.resampling_pupil.gamma % 2 == 0
    except AssertionError:
      self.logger.warning(" Pupil gamma should be even. Could produce " + \
        "unexpected results...")    
      
    try:
      assert isPowerOfTwo(self.resampling_pupil.sampling) == True
    except AssertionError:
      self.logger.critical(" Pupil sampling should be a power of two!")
      exit(0)
      
    # Take a copy of the [resampling_pupil] instance and construct a new image 
    # instance for this wavelength. 
    #
    # The only thing that is different between the different wavelength 
    # instances will be the pixel scales in both the pupil and image planes 
    # as the spatial/angular size of a resolution element. We need to rescale
    # the data so it has the same spatial pixel scale as [resampled_im] and 
    # then move back to the conjugate pupil plane.
    #
    # We must take a copy of the [resampling_pupil], otherwise this will be 
    # overwritten when we resample. Resampling is an in-place operation.
    #
    this_pupil = self.resampling_pupil.copy()
    this_im = this_pupil.toConjugateImage(wave, self.preoptics_reimager, 
      verbose=True)
    this_im.resample(self.resampling_im.p_pixel_scale, verbose=True)
    this_pupil = this_im.toConjugatePupil(verbose=True)

    # Assign this as the composite image. This instance will be used to 
    # construct the final image from a series of slices.
    #
    this_composite_image = this_im

    # Slice the field up.
    #
    # We use only a central section of the reimaged slicing FoV.
    #
    npix_slice_width = int(\
      self.instrument.slicer_dimensions_physical_active[0] / \
      this_im.p_pixel_scale)
    npix_slice_height = int(\
      self.instrument.slicer_dimensions_physical_active[1] / \
      this_im.p_pixel_scale / self.instrument.n_slices)

    y_s = int((this_im.data.shape[0]/2.) - ((npix_slice_height *\
      self.instrument.n_slices)/2.))
    x_s = int((this_im.data.shape[1]/2) - (npix_slice_width/2))
    x_region = []
    y_region = []
    for s in range(self.instrument.n_slices):
      x_region.append((x_s, x_s + npix_slice_width))
      y_region.append(((s*npix_slice_height)+y_s, (s*npix_slice_height) + \
        y_s + npix_slice_height))

    # Get entrance slit fields corresponding to each slice.
    #
    # This is returned as an array of fields for each slice, we need to flatten 
    # this out as we've only requested 1 field per slit.
    #
    fields = self.instrument.getEntranceSlitFields(
      n_fields_per_slitlet=1,
      n_spectrographs=self.instrument.n_spectrographs)
    tmp = []
    for f in fields:
      tmp.append(f[0])
    fields = tmp

    # Get WFE maps for each component as requested.
    #
    if self.cfg['SIM_ADD_COLLIMATOR_WFE']:
      wfe_col_d, wfe_col_h = self.zspec.collimator.getWFE(fields, float(wave),
        sampling=self.cfg['PUPIL_WFE_MAP_SAMPLING'])        

    if self.cfg['SIM_ADD_CAMERA_WFE']:
      cam_OA = self.zspec.collimator.getOA(fields, float(wave), verbose=False)
      wfe_cam_d, wfe_cam_h = self.zspec.camera.getWFE(cam_OA, float(wave),
        sampling=self.cfg['PUPIL_WFE_MAP_SAMPLING']) 

    # For each slice ..
    #
    slices = []
    s_idx = 1
    for this_x_region, this_y_region in zip(x_region, y_region):
      self.logger.debug(" Considering slice " + str(s_idx) + " of " + \
        str(self.instrument.n_slices))
      
      # Move to image plane and mask outside the slice region
      #
      im = this_pupil.toConjugateImage(wave, self.preoptics_reimager, 
        verbose=False)
      this_slice_im = im.asRegion(this_x_region[0], this_x_region[1], \
        this_y_region[0], this_y_region[1], verbose=True)

      # Move back to pupil plane.
      #
      this_slice_pupil = this_slice_im.toConjugatePupil()

      # Add camera WFE
      #
      if self.cfg['SIM_ADD_COLLIMATOR_WFE']:
        if verbose:
          self.logger.debug(" Adding collimator WFE.")
        WFE_pupil_diameter = self.zspec.collimator.getEXPD(wave)
        WFE_sampling = wfe_col_h[s]['SAMPLING'][0]
        wfe = this_slice_pupil.addWFE(WFE_pupil_diameter, WFE_sampling, 
          wfe_col_d[s])

      # Add camera WFE
      #
      if self.cfg['SIM_ADD_CAMERA_WFE']:
        if verbose:
          self.logger.debug(" Adding camera WFE.")
        WFE_pupil_diameter = self.zspec.camera.getEXPD(wave)
        WFE_sampling = wfe_cam_h[s]['SAMPLING'][0]
        wfe = this_slice_pupil.addWFE(WFE_pupil_diameter, WFE_sampling, 
          wfe_cam_d[s])

      # Move back to image plane and overwrite this slice region.
      #
      this_slice_im = this_slice_pupil.toConjugateImage(wave, 
        self.preoptics_reimager, verbose=False)
      this_composite_image.setRegionData(this_x_region[0], this_x_region[1], \
        this_y_region[0], this_y_region[1], this_slice_im.data)

      s_idx+=1
    return this_composite_image
    

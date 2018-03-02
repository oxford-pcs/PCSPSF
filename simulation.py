import pylab as plt

from ifu_builder.ifu import IFU_SWIFT
from products import cube
from util import isPowerOfTwo

class sim():
  def __init__(self, logger, plotter, resampling_im, resampling_pupil, nwaves, 
               preoptics_reimager, spec, cfg):
    self.logger = logger
    self.plotter = plotter
    self.resampling_im = resampling_im
    self.resampling_pupil = resampling_pupil
    self.nwaves = nwaves
    self.preoptics_reimager = preoptics_reimager
    self.spec = spec
    self.cfg = cfg

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
    # as the angular size of a resolution element is equal to lambda / D. 
    # We need to rescale the data so it has the same pixel scale as 
    # [resampled_im] and then move back to the conjugate pupil plane.
    #
    # We must take a copy of the [resampling_pupil], otherwise this will be 
    # overwritten when we resample. Resampling is an in-place operation.
    #
    this_pupil = self.resampling_pupil.copy()
    this_im = this_pupil.toConjugateImage(wave, self.preoptics_reimager, 
      verbose=True)
    this_im.resample(self.resampling_im.p_pixel_scale, 
                     self.resampling_im.p_detector_FOV, verbose=True)
    this_pupil = this_im.toConjugatePupil(verbose=True)

    # Assign this as the composite image. This instance will be used to 
    # construct the final image from a series of slices.
    #
    this_composite_image = this_im

    # Build the IFU.
    #
    ifu = IFU_SWIFT(self.cfg['SIM_PREOPTICS_CFG_NAME'], 
      self.cfg['SIM_SLICER_CFG_NAME'], self.cfg['SIM_SLIT_CFG_NAME'], 
      config_dir=self.cfg['SIM_IFU_CONFIGS_DIR_PATH'])

    # Slice the field up.
    #
    # This conjugate image plane is the image plane at the slicer. 
    #
    # We use only a central section of this FoV defined from the number of 
    # slices in the slicer stack and the number of resolution elements per 
    # slice. 
    #
    print ifu.getEntranceSlitFields(44)
    exit(0)

    npix_slice_height = int(self.resampling_pupil.gamma * \
      self.cfg['SLICE_RESEL_PER_SLICE']) 
    npix_slice_width = int(self.resampling_pupil.gamma * \
      self.cfg['SLICE_RESEL_PER_SLICE'] * \
      pattern_data['n_spaxels_per_slitlet'] * \
      pattern_data['stack_wh_aspect_ratio'])

    x_s = int((this_pupil.data.shape[1]/2) - (npix_slice_width/2))
    y_s = int((this_pupil.data.shape[0]/2.) - (npix_slice_height/2.))
    x_region = []
    y_region = []
    for s in range(pattern_data['n_slitlets']):
      x_region.append(range(x_s, x_s + npix_slice_width, 1))    #FIXME
      y_region.append(range(y_s, y_s + npix_slice_height, 1))   #FIXME

    # Get field points corresponding to each slice.
    #
    slit_pattern = slit(self.slits_file, self.slits_name)
    pattern_data = slit_pattern.cfg['pattern_data']
    fields = slit_pattern.getFieldsFromSlitPattern(
      nfields=pattern_data['n_slitlets'])

    # Get WFE maps for each component as requested.
    #
    if self.cfg['SIM_ADD_COLLIMATOR_WFE']:
      wfe_cam_d, wfe_cam_h = self.spec.collimator.getWFE(fields, float(wave),
        sampling=self.cfg['PUPIL_WFE_MAP_SAMPLING'])        

    if self.cfg['SIM_ADD_CAMERA_WFE']:
      cam_OA = self.spec.collimator.getOA(fields, float(wave), verbose=False)
      wfe_cam_d, wfe_cam_h = self.spec.camera.getWFE(cam_OA, float(wave),
        sampling=self.cfg['PUPIL_WFE_MAP_SAMPLING']) 

    # Process for each field point.
    #
    slices = []
    for field_x, field_y in zip(x_region, y_region):
      print field_x, field_y
      exit(0)
      this_slice_x_s = x_region[0]
      this_slice_x_e = x_region[-1]
      this_slice_y_s = y_region[s*npix_per_slice]
      this_slice_y_e = this_slice_y_s + npix_per_slice
      this_slice_region = [(this_slice_x_s, this_slice_x_e),
                           (this_slice_y_s, this_slice_y_e)]
 
      # Move to image plane and take a slice.
      #
      im = this_pupil.toConjugateImage(wave, self.preoptics_reimager, 
        verbose=False)
      this_slice_im = im.toRegion(this_slice_region, verbose=True)

      # Move back to pupil plane.
      #
      this_slice_pupil = this_slice_im.toConjugatePupil()

      # TODO: Add collimator WFE and resample these maps to 
      # the [resampling_pupil] plate scale.

      # Add camera WFE
      #
      if self.cfg['SIM_ADD_CAMERA_WFE']:
        self.logger.debug(" Adding camera WFE.")
        WFE_pupil_diameter = self.spec.camera.getENPD(wave)
        wfe = this_slice_pupil.addWFE(WFE_pupil_diameter, wfe_cam_d[s], 
          wfe_cam_h[s])

      # Move back to image plane.
      #
      this_slice_im = this_slice_pupil.toConjugateImage(wave, 
        self.preoptics_reimager, verbose=False)
      this_composite_image.setRegionData(this_slice_region, this_slice_im.data)
          
    return this_composite_image
    

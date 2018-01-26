import pylab as plt

from products import cube
from util import isPowerOfTwo
from camera import pcamera

class sim():
  def __init__(self, logger, plotter, resampling_im, resampling_pupil, nwaves, 
               camera, cfg, args):
    self.logger = logger
    self.plotter = plotter
    self.resampling_im = resampling_im
    self.nwaves = nwaves
    self.camera = camera
    self.resampling_pupil = resampling_pupil
    self.nslices = cfg['SLICE_NUMBER_OF']
    self.resel_per_slice = cfg['SLICE_RESEL_PER_SLICE']
    self.resample_to_wavelength = cfg['PUPIL_RESAMPLE_TO_WAVELENGTH']
    self.ADD_CAM_WFE = args.caw
    self.ADD_COL_WFE = args.cow
    
  def run(self, wave, verbose=True): 
    
    self.physical_pupil_diameter = self.resampling_pupil.physical_pupil_diameter
    self.pupil_sampling = self.resampling_pupil.sampling
    self.pupil_gamma = self.resampling_pupil.gamma
    
    #  Sanity checks.
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
    # instance for this wavelength. Although the physical data will be 
    # identical, the resultant pupil and image will have different pixel scales. 
    # We need to rescale these so they have the same pixel scale as 
    # [resampled_im]. We can then move back to the conjugate pupil plane.
    #
    # We must take a copy of the [resampling_pupil], otherwise this will be 
    # overwritten when we resample.
    #
    this_pupil = self.resampling_pupil.copy()
    this_im = this_pupil.toConjugateImage(wave, self.camera, verbose=True)
    this_im.resample(self.resampling_im.p_pixel_scale, 
                     self.resampling_im.p_detector_FOV, verbose=True)
    this_pupil = this_im.toConjugatePupil(verbose=True)
    
    # Assign a composite image. This instance will be used to construct the 
    # final image from a series of slices.
    #
    this_composite_image = this_im
    
    # Slice the field up.
    #
    # The conjugate image plane will generally have a rather large FoV (due 
    # to the large sampling grid). We use only a central section of this FoV 
    # defined from the number of slices in the slicer stack and the number of 
    # resolution elements per slice. Since this only defines the size of one 
    # axis, we assume that the slicer stack is square.
    #
    npix_per_slice = int(self.resampling_pupil.gamma * self.resel_per_slice) 
    npix_slicer_stack = npix_per_slice * self.nslices

    x_s = int((this_pupil.data.shape[1]/2) - (npix_slicer_stack/2))
    y_s = int((this_pupil.data.shape[0]/2.) - (npix_slicer_stack/2.))
    x_region = range(x_s, x_s + npix_slicer_stack, 1)
    y_region = range(y_s, y_s + npix_slicer_stack, 1)
    
    slices = []
    for s in range(self.nslices):	
      this_slice_x_s = x_region[0]
      this_slice_x_e = x_region[-1]
      this_slice_y_s = y_region[s*npix_per_slice]
      this_slice_y_e = this_slice_y_s + npix_per_slice
      this_slice_region = [(this_slice_x_s,
                            this_slice_x_e),
                           (this_slice_y_s,
                            this_slice_y_e)]
 
      # Move to image plane and take a slice.
      #
      im = this_pupil.toConjugateImage(wave, self.camera, verbose=False)
      this_slice_im = im.toSlice(this_slice_region, verbose=True)

      # Move to pupil plane.
      #
      this_slice_pupil = this_slice_im.toConjugatePupil()

      # TODO: ADD WFE

      # Move back to image plane.
      #
      this_slice_im = this_slice_pupil.toConjugateImage(wave, self.camera, 
                                                        verbose=False)
      this_composite_image.setRegionData(this_slice_region, this_slice_im.data)
          
    return this_composite_image
    

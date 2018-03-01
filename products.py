import os
import ntpath
import decimal
import copy

import numpy as np
import pyfits

from util import sf, resample2d

class cube():
  '''
    A 3D datacube.
  '''
  def __init__(self, logger, dshape):
    self.logger = logger
    self.dshape = dshape
    self.images = []
  
  def append(self, im):
    '''
      Add an image instance [image] to the cube.
    '''
    self.images.append(im)
    
  def update(self, im, index):
    '''
      Update image at index [index] of the cube with image [im].
    '''
    self.images[index] = im  

  def write(self, args, cfg, xtra_header_keys, clobber=True):
    '''
      Make a hard copy of the datacube.
    '''
    header = pyfits.Header()    
    
    if os.path.exists(args.o):
      if clobber == True:
        os.remove(args.o)
      else:
        self.logger.debug("Output filename already exists and clobber" + \
          " is not set.")
        exit(0)
    self.logger.debug(" Writing output to " + str(args.o) + ".")
  
    data = []
    for im in self.images:
      if im.data.shape != self.dshape:
        self.logger.debug("Image data shape is not consistent with cube " + \
          " [dshape].")
        exit(0)
      data.append(im.getAmplitude())
    data = np.array(data)
    
    pyfits.writeto(args.o, data, header)


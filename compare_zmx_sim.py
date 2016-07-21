'''
  This program compares unsliced simulated output to the equivalent from zemax.
  
  The procedure for generating the two files is as follows:
  
  1) Ensure the parameters are correct (wfno etc.) in default.ini
  2) Generate an output file with the -f flag, e.g. sampling 256, gamma 8. Ensure verbose (-v) option is set
  3) Get per pixel spatial resolution and size of detector (micron)
  4) In Zemax, go to Analysis > FFT PSF
  5) In the settings, adjust the image delta and sampling to match both the pixel spatial resolution and the size of the detector
  6) Output the file (Window > Text)
  7) Run this program with text file as first argument (from 6) and output simulated file (from 2) as second
'''

import logging

import pylab as plt
import pyfits
import numpy as np
import sys

from zmx_parser import zfftpsf

# setup logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)s:%(asctime)s:%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

z = zfftpsf(sys.argv[1], logger, True)
z.parse()
d_zemax = z.getData()
d_zemax = np.roll(d_zemax, 1, axis=1)

hdulist = pyfits.open(sys.argv[2])
d_sim = hdulist[0].data

d_sim = d_sim**2			# because zemax "linear" data is the power spectrum!

diff = d_sim - d_zemax

plt.figure()
plt.subplot(131)
plt.imshow(d_zemax, interpolation="None")
plt.colorbar()
plt.subplot(132)
plt.imshow(d_sim, interpolation="None")
plt.colorbar()
plt.subplot(133)
plt.imshow(diff, interpolation="None")
plt.colorbar()
plt.tight_layout()
plt.show()
import sys
import os

import imageio
import pylab as plt
import pyfits
import numpy as np

cube = pyfits.open(sys.argv[1])

n_slices = cube[0].data.shape[0]
dim_1 = cube[0].data.shape[1]
dim_2 = cube[0].data.shape[2]

dmin = np.min(cube[0].data)
dmax = np.max(cube[0].data)

images = []
for s in range(n_slices):
  fname = str(s) + ".png"
  plt.imshow(cube[0].data[s], interpolation='none', vmin=dmin, vmax=dmax)
  plt.colorbar()
  plt.savefig(fname)
  plt.close()
  images.append(imageio.imread(fname))

imageio.mimsave('movie.gif', images)

for s in range(n_slices):
  fname = str(s) + ".png"
  os.remove(fname)

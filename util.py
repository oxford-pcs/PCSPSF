import numpy as np
from scipy.interpolate import RectBivariateSpline

def sf(fig, n):
  format = '%.' + str(n) + 'g'
  return '%s' % float(format % float(fig))

def resample2d(i_data, i_s, i_e, i_i, o_s, o_e, o_i):
  # set up original grid
  grid_pre_rebin = np.arange(i_s, i_e, i_i)
  G = RectBivariateSpline(grid_pre_rebin, grid_pre_rebin, i_data, kx=3, ky=3)

  # evaluate at new sampling and to limits 
  grid_x, grid_y = np.mgrid[o_s:o_e:o_i, o_s:o_e:o_i]
  return G.ev(grid_x, grid_y)
        
        

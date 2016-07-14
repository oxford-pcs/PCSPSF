import pylab as plt
import numpy as np

class plotter():
  def __init__(self):
    self.NPLOTS = 0		# number of plots
    self.PLOTDATA = []		# plot information
    
  def _addImagePlot(self, title, data, cb=True, extent=None, xl=None, yl=None, overplot=False):
    self.PLOTDATA.append({"title": title, "data": data, "cb": cb, "type": "im", "extent": extent, "xl": xl, "yl": yl, "overplot": overplot})
    if not overplot:
      self.NPLOTS = self.NPLOTS+1
      
  def _addScatterPlot(self, title, y, x=None, color='w', ls='--', cb=True, xl=None, yl=None, xr=None, yr=None, overplot=False):  
    self.PLOTDATA.append({"title": title, "x": x, "y": y, "color": color, "ls": ls, "cb": cb, "type": "scatter", "xl": xl, "yl": yl, "xr": xr, "yr": yr, "overplot": overplot})
    if not overplot:
      self.NPLOTS = self.NPLOTS+1
      
  def _reset(self):
    self.__init__()
    
  def draw(self, nrows, ncols):
    plt.figure()
    nplot = 1
    for d in self.PLOTDATA:
      if not d['overplot']:
        plt.subplot(ncols,nrows,nplot)
        plt.title(d['title'], fontsize=12)
        nplot = nplot+1
      if d['type'] == 'im':
	plt.imshow(d['data'], extent=d['extent'], interpolation='none')
	if d['xl'] is not None:
	   plt.xlabel(d['xl'], fontsize=12)
	if d['yl'] is not None:
	   plt.ylabel(d['yl'], fontsize=12)	   
	if d['cb']:
          plt.colorbar()
      if d['type'] == 'scatter':
	if d['x'] == None:
	  plt.plot(d['y'], color=d['color'], linestyle=d['ls'])
	else:
	  plt.plot(d['x'], d['y'], color=d['color'], linestyle=d['ls'])
	if d['xl'] is not None:
	   plt.xlabel(d['xl'], fontsize=12)
	if d['yl'] is not None:
	   plt.ylabel(d['yl'], fontsize=12)
	if d['xr'] is not None:
	   plt.xlim(d['xr'])
	if d['yr'] is not None:
	   plt.ylim(d['yr'])
    plt.tight_layout()
    plt.show()
    self._reset()
    


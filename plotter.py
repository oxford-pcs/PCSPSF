import pylab as plt
import numpy as np

class plotter():
  def __init__(self):
    self.NPLOTS = 0		# number of plots
    self.PLOTDATA = []		# plot information
  def _addImagePlot(self, title, data, cb=True, extent=None, xl=None, yl=None):
    self.PLOTDATA.append({"title": title, "data": data, "cb": cb, "type": "im", "extent": extent, "xl": xl, "yl":yl})
    self.NPLOTS = self.NPLOTS+1
  def _addScatterPlot(self, title, y, x=None, cb=True):  
    self.PLOTDATA.append({"title": title, "x": x, "y": y, "cb": cb, "type": "scatter"})
    self.NPLOTS = self.NPLOTS+1
  def draw(self, nrows, ncols):
    plt.figure()
    for p_idx, d in enumerate(self.PLOTDATA):
      plt.subplot(ncols,nrows,p_idx+1)
      plt.title(d['title'], fontsize=12)
      if d['type'] == 'im':
	plt.imshow(d['data'], extent=d['extent'])
	if d['xl'] is not None:
	   plt.xlabel(d['xl'], fontsize=12)
	if d['yl'] is not None:
	   plt.ylabel(d['yl'], fontsize=12)	   
	if d['cb']:
          plt.colorbar()
      if d['type'] == 'scatter':
	if d['x'] == None:
	  plt.plot(d['y'])
	else:
	  plt.plot(d['x'], d['y'])
    plt.tight_layout()
    plt.show()
    


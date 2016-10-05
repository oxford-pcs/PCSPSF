import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from util import resample2d

class EditablePupil:
    def __init__(self, data):
      self.print_blurb()
      
      self.fig = plt.figure()
      self.gs = gridspec.GridSpec(2, 3)
      
      self.data = data 									# data (complex)
      self.cx = 0									# cursor x
      self.cy = 0									# cursor y
      
      self.display_types = ['magnitude', 'phase', 'real', 'complex']			# list of available display modes
      self.current_display = self.display_types[0]					# current display mode
    
      self.ax1_data = np.abs(data)**2   						# initial data (magnitude) 
      self.ax2_data = ([0, 0], [0, 0])							# initial profile 
      self.ax3_data = ([0, 0], [0, 0])							# initial profile
     
      self.ax1 = self.imshow(self.ax1_data, "magnitude", gs_win=self.gs[:,:-1])		# data plot
      self.lx = self.ax1.axhline(color='w')						# data xhair, no init
      self.ly = self.ax1.axvline(color='w')						# data xhair, no init
      self.ax2 = self.plot(*self.ax2_data, gs_win=self.gs[0,-1:])			# profile plot
      self.ax3 = self.plot(*self.ax3_data, gs_win=self.gs[1,-1:])			# profile plot

      self.next_is_inverse_fftshift = False
      self.next_is_inverse_fft2 = False
      
      self.draw_data(oper="")
        
    def connect(self):
      self.cidkeypress = plt.connect('key_press_event', self.on_press)
      self.cidmousemove = plt.connect('motion_notify_event', self.mouse_move)
             
    def draw_data(self, oper):
      self.populate_ax1_data()
      self.imshow(self.ax1_data, label=oper, title=self.current_display, gs_win=self.gs[:,:-1])
      plt.draw()
      
    def draw_profiles_and_xhair(self):
      x = np.arange(*self.ax1.get_ylim())
      y = self.ax1_data[:, self.cx]
      self.ax2.cla()
      self.ax2.plot(x, y)
      self.ax2.set_xlim([np.min(x), np.max(x)])
      
      x = np.arange(*self.ax1.get_xlim())
      y = self.ax1_data[self.cy, :]
      self.ax3.cla()
      self.ax3.plot(x, y)
      self.ax3.set_xlim([np.min(x), np.max(x)])   
      
      self.lx.set_ydata(self.cy)
      self.ly.set_xdata(self.cx)
      plt.draw()
	
    def imshow(self, data, label="", title="", gs_win=None, cb=True):
      if gs_win is None:
	gs_win=self.gs[:,:]
      ax = plt.subplot(gs_win)
      ax.annotate(label, (0, 0), xycoords="axes points", fontsize=15, color="w")
      plt.imshow(data)
      plt.title(title)
      if cb:
        plt.colorbar()
      plt.tight_layout()
      plt.xlim([0,data.shape[1]])
      plt.ylim([0,data.shape[1]])
      return plt.gca()
    
    def mouse_move(self, event):
      if not event.inaxes:
	  return
      self.cx, self.cy = event.xdata, event.ydata
      self.draw_profiles_and_xhair()
          	
    def on_press(self, event):
      oper = "unrecognised"
      if event.key == "1":
	if not self.next_is_inverse_fft2:
	  oper = "fft2"
	  self.data = np.fft.fft2(self.data)
	else:
	  oper = "ifft2"
	  self.data = np.fft.ifft2(self.data)
	self.next_is_inverse_fft2 = not self.next_is_inverse_fft2
	self.next_is_inverse_fftshift = False     
      elif event.key == "2":
	if not self.next_is_inverse_fftshift:
	  oper = "fftshift"
	  self.data = np.fft.fftshift(self.data)
	  self.next_is_inverse_fftshift = True
	else:
	  oper = "ifftshift"
	  self.data = np.fft.ifftshift(self.data)
      elif event.key == "pagedown":
	oper = "switched display mode"
	idx = self.display_types.index(self.current_display)-1
        idx = len(self.display_types)-1 if idx < 0 else idx   	 	# wraparound
	self.current_display = self.display_types[idx]
      elif event.key == "pageup":
	oper = "switched display mode"
	idx = self.display_types.index(self.current_display)+1
        idx = 0 if idx > len(self.display_types)-1 else idx   	 	# wraparound
	self.current_display = self.display_types[idx]
      self.draw_data(oper=oper)
        
    def plot(self, x, y, gs_win=None):
      if gs_win is None:
	gs_win=self.gs[:,:]
      ax = plt.subplot(gs_win)
      plt.plot(x, y)
      return plt.gca()
      
    def populate_ax1_data(self): 
      if self.current_display == 'magnitude':
	self.ax1_data = np.abs(self.data)**2
      elif self.current_display == 'phase':
	self.ax1_data = np.angle(self.data)
      elif self.current_display == 'real':
	self.ax1_data = np.real(self.data)
      elif self.current_display == 'complex':
	self.ax1_data = np.imag(self.data)**2
	
    def print_blurb(self):
      print 
      print "List of commands:"
      print
      print "1:\tfft2/ifft2"
      print "2:\tfftshift/ifftshift"
      print "pgup:\tcycle display mode (up)"
      print "pgdn:\tcycle display mode (down)"
      print      
      
if __name__=="__main__":
  
  sampling=256
  # Construct pupil
  y, x = np.ogrid[-sampling/2:sampling/2, -sampling/2:sampling/2]
  mask = x*x + y*y <= ((sampling-1)/2)*((sampling-1)/2)
  data = np.zeros((sampling, sampling), dtype='complex')
  data[mask] = 1 + 0j

  dr = EditablePupil(data)
  dr.connect()

  plt.show()
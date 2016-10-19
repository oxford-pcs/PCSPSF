import sys
import copy

from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

from util import resample2d
from ui import ui
  
class editable():
  '''
    This editable class allows a user to manipulate pupil data.
    
    An editable requires a qt4 user interface with draw_ui() method and appropriate
    buttons/canvas placeholder.
  '''
  def __init__(self, ui, data):
    self.fig = plt.figure()						# matplotlib fig
    self.canvas = FigureCanvas(self.fig)				# attach fig to canvas
    
    self.modes = ['inspect', 						# interaction modes
		  'resize', 						#
		  'resize_inprogress', 					#
		  'mask_re',						#
		  'mask_ci',						#
		  'mask_inprogress']					# 
    self.current_mode = None						# this will be set on .connect_ call
    
    self.cidmousebuttonpress 		= None				# init event ids, need to keep track for disconnects
    self.cidmousebuttonrelease 	= None					#
    self.cidmousemove			= None				#
    self.cidkeypress			= None				#
    
    self.next_is_inverse_fft2 = False					# so we know whether to use inverse function or not
    self.next_is_inverse_fftshift = False				#
    
    self.ax1_background = None						# used in animations so we don't have to redraw canvas from data
    
    self.ui = ui							# instantiate and draw qt4 ui
    self.ui.draw_ui(self.canvas)					#
      
    self.gs = gridspec.GridSpec(2, 3)					# split matplotlib fig. into gridspec subplots
    self.ax1 = self.fig.add_subplot(self.gs[:,:-1])			#
    self.ax1_img = None							#
    self.ax1_cb = None							#
    self.ax2 = self.fig.add_subplot(self.gs[0,-1:])			#
    self.ax2_line = None						#
    self.ax3 = self.fig.add_subplot(self.gs[1,-1:])			#
    self.ax3_line = None						#

    self.data_in = np.copy(data)					# used to revert
    self.data = data 							# data (complex)
    
    self.cx = 0								# cursor event x
    self.cy = 0								# cursor event y
    self.xx = 0								# xhair x
    self.xy = 0								# xhair y
      
    self.ax1_data = np.zeros(shape=data.shape)  			# initial data (will be populated later)
    self.ax2_data = ([0, 1], [0, 0])					# initial profile 
    self.ax3_data = ([0, 1], [0, 0])					# initial profile
    
    self.display_types = ['magnitude', 					# list of available display modes
			  'phase', 					#
			  'real', 					#
			  'imag']					# 
    self.current_display = 0						# current display mode      
    self.data_scales = ['linear', 					# list of available data scaling modes
			'log', 						#
			'99.5%']					# 

    self.current_data_scaling = 0					# current data scaling mode
    
    self.current_display_subsection = [(0,0),np.shape(data)]		# current window (used in resizing)

    self._populate_ax1_data()
    self._draw_data()							# draw img data
    self._draw_profiles()						# draw x/y profiles
    
    self.fig.canvas.draw()						# call initial draw

  def _action_fft(self):
    ''' callable for fft/ifft button click '''
    if not self.next_is_inverse_fft2:
      self.data = np.fft.fft2(self.data)
    else:
      self.data = np.fft.ifft2(self.data)
    self.next_is_inverse_fft2 = not self.next_is_inverse_fft2
    self.next_is_inverse_fftshift = False 
    self._populate_ax1_data()
    self._draw_data()
    self._draw_profiles()
    self._draw_xhair()
    self.ui.change_message_text("fft/ifft", critical=False)
    self.fig.canvas.draw()
    
  def _action_fftshift(self):
    ''' callable for fftshift/ifftshift button click '''
    if not self.next_is_inverse_fftshift:
      self.data = np.fft.fftshift(self.data)
      self.next_is_inverse_fftshift = True
    else:
      self.data = np.fft.ifftshift(self.data)
    self._populate_ax1_data()
    self._draw_data()
    self._draw_profiles()
    self._draw_xhair()
    self.ui.change_message_text("fftshift/ifftshift", critical=False)
    self.fig.canvas.draw() 
      
  def _control_scaling_next(self):
    ''' switch data scaling (next) '''
    idx = self.current_data_scaling-1
    idx = len(self.data_scales)-1 if idx < 0 else idx   	 # wraparound
    self.current_data_scaling = idx
    self._populate_ax1_data()
    self._draw_data()
    self._draw_profiles()
    self._draw_xhair()
    self.ui.change_message_text("changed data scaling (" + self.data_scales[self.current_data_scaling] + ")", critical=False)
    self.fig.canvas.draw()

  def _control_scaling_prev(self):
    ''' switch data scaling (prev) '''
    idx = self.current_data_scaling+1
    idx = 0 if idx > len(self.data_scales)-1 else idx   	 # wraparound
    self.current_data_scaling = idx
    self._populate_ax1_data()
    self._draw_data()
    self._draw_profiles()
    self._draw_xhair()
    self.ui.change_message_text("changed data scaling (" + self.data_scales[self.current_data_scaling] + ")", critical=False)
    self.fig.canvas.draw()  	
      
  def _control_display_next(self):
    ''' switch display mode (next) '''
    idx = self.current_display-1
    idx = len(self.display_types)-1 if idx < 0 else idx   	 # wraparound
    self.current_display = idx
    self._populate_ax1_data()
    self._draw_data()
    self._draw_profiles()
    self._draw_xhair()
    self.ui.change_message_text("changed display mode (" + self.display_types[idx] + ")", critical=False)
    self.fig.canvas.draw()

  def _control_display_prev(self):
    ''' switch display mode (prev) '''
    idx = self.current_display+1
    idx = 0 if idx > len(self.display_types)-1 else idx   	 # wraparound
    self.current_display = idx
    self._populate_ax1_data()
    self._draw_data()
    self._draw_profiles()
    self._draw_xhair()
    self.ui.change_message_text("changed display mode (" + self.display_types[idx]  + ")", critical=False)
    self.fig.canvas.draw()
    
  def _control_reset_view(self):
    ''' reset view i.e. no zoom '''
    self.current_display_subsection = [(0,0),np.shape(self.data)]	
    self._populate_ax1_data()
    self._draw_data()
    self._draw_profiles()
    self._draw_xhair()
    self.ui.change_message_text("reset view", critical=False)
    self.fig.canvas.draw()
    
  def _control_reset_data(self):
    ''' reset to original state '''
    self.data = np.copy(self.data_in)
    
    self.cx = 0
    self.cy = 0
    self.xx = 0
    self.xy = 0
    
    self.ax1_data = np.zeros(shape=self.data.shape)
    self.ax2_data = ([0, 1], [0, 0])
    self.ax3_data = ([0, 1], [0, 0])
    
    self.current_display = 0
    self.current_data_scaling = 0
    self.current_display_subsection = [(0,0),np.shape(self.data)]
    
    self._populate_ax1_data()
    self._draw_data()
    self._draw_profiles()
    self.ui.change_message_text("reset", critical=False)
    self.fig.canvas.draw()
	  
  def _disconnect_all_connects(self):
    if self.cidmousebuttonpress is not None:
      self.fig.canvas.mpl_disconnect(self.cidmousebuttonpress)
    if self.cidmousebuttonrelease is not None:
      self.fig.canvas.mpl_disconnect(self.cidmousebuttonrelease)
    if self.cidmousemove is not None:
      self.fig.canvas.mpl_disconnect(self.cidmousemove)     
    if self.cidkeypress is not None:
      self.fig.canvas.mpl_disconnect(self.cidkeypress)    
    
  def _draw_data(self):
    ''' draw view data, incl. colorbars '''
    self.ax1_img = self._imshow(self.ax1, 
				self.ax1_data, 
				self.current_display_subsection, 
				title=self.display_types[self.current_display])	# populate view
    self.lx = self.ax1.axhline(-1, color='w')						# init xhair
    self.ly = self.ax1.axvline(-1, color='w')						# init xhair
    if not self.ax1_cb:								# add colorbar
      self.ax1_cb = plt.colorbar(self.ax1_img, ax=self.ax1)
    else:
      self.ax1_cb.set_clim(vmin=np.min(self.ax1_data), vmax=np.max(self.ax1_data))
      self.ax1_cb.draw_all()

  def _draw_profiles(self):
    ''' draw x/y slice profiles '''
    xr = (0,self.ax1_data.shape[1])
    x = np.arange(*xr)
    y = self.ax1_data[self.cy-self.current_display_subsection[0][1], x]
    self.ax2_line = self._plot(self.ax2, x, y, line=self.ax2_line)
    
    xr = (0,self.ax1_data.shape[0])
    x = np.arange(*xr)
    y = self.ax1_data[x, self.cx-self.current_display_subsection[0][0]]
    self.ax3_line = self._plot(self.ax3, x, y, line=self.ax3_line) 
    
  def _draw_xhair(self):
    ''' draw crosshair '''
    self.lx.set_ydata(self.xy)
    self.ly.set_xdata(self.xx) 
    
  def _imshow(self, ax, data, subsection, title=""):
    ''' callable by _draw_data(), draws [data] to the axis [ax] '''
    ax.cla()
    data_min = np.min(data)
    data_max = np.max(data)
    if np.isclose(data_min, data_max):
      data_min = 0
      if data_max == 0:
	data_max = 1
    extent = (subsection[0][0], subsection[1][0], subsection[0][1], subsection[1][1])
    img = ax.imshow(data, interpolation='none', origin='lower', vmin=data_min, vmax=data_max, extent=extent)
    ax.set_title(title)
    ax.set_xlim(extent[0:2])
    ax.set_ylim(extent[2:])
    return img
  
  def _inspect(self, event):
    ''' inspect mode: draw x/y slice profiles and crosshair '''
    if event.inaxes is None:
      return
    elif not str(event.inaxes.get_axes()) == str(self.ax1.get_axes()):	# is this self.ax1?
      return
    self.cx, self.cy = event.xdata, event.ydata
    self.xx, self.xy = self.cx, self.cy
    self._draw_profiles()
    self._draw_xhair()  
    self.fig.canvas.draw() 
	
  def _mask_ci_move(self, event):
    if event.inaxes is None:
      return
    elif not str(event.inaxes.get_axes()) == str(self.ax1.get_axes()):	# is this self.ax1?
      return
    self.cx, self.cy = event.xdata, event.ydata
    if self.modes[self.current_mode] == 'mask_inprogress':
      r = self._set_circ_radius(self.mask, self.cx, self.cy)
      self.canvas.restore_region(self.ax1_background)
      self.ax1.draw_artist(self.mask)
      self.fig.canvas.update()
      self.fig.canvas.flush_events()
    
  def _mask_ci_press(self, event):
    if event.inaxes is None:
      return
    elif not str(event.inaxes.get_axes()) == str(self.ax1.get_axes()):	# is this self.ax1?
      return
    self.cx, self.cy = event.xdata, event.ydata
    
    try:									# init/reset patch used for masking
      self.mask.remove()
    except (AttributeError, ValueError):
      pass
    self.mask = self.ax1.add_patch(patches.Circle(			
      (0, 0),
      0,
      color='w',
      linestyle='dashed',
      fill=False)
    )
    self.ax1_background = self.canvas.copy_from_bbox(self.ax1.bbox)   

    ox, oy = self._set_circ_origin(self.mask, self.cx, self.cy)

    self.ui.change_message_text("masking")   
    self._switch_current_mode('mask_inprogress')
    
  def _mask_ci_release(self, event):
    self._switch_current_mode('mask_ci')
    self.fig.canvas.draw()							# to clear
    if event.inaxes is None:
      return
    elif not str(event.inaxes.get_axes()) == str(self.ax1.get_axes()):	# is this self.ax1?
      return
    self.cx, self.cy = event.xdata, event.ydata
    if int(self.cx) == self.mask.center[0] or int(self.cy) == self.mask.center[1]:
      self.ui.change_message_text("not a valid mask", critical=True)   
      return
    r = self._set_circ_radius(self.mask, self.cx, self.cy)
    o = self.mask.center
    self.ui.button_mask_param1.setText(str(o[0]))
    self.ui.button_mask_param2.setText(str(o[1]))
    self.ui.button_mask_param3.setText(str(r))
    self.ui.button_mask_param4.setText(str("0"))
    self.ui.change_message_text("adjust/choose a value for mask and click apply")         
	
  def _mask_re_move(self, event):
    if event.inaxes is None:
      return
    elif not str(event.inaxes.get_axes()) == str(self.ax1.get_axes()):	# is this self.ax1?
      return
    self.cx, self.cy = event.xdata, event.ydata
    if self.modes[self.current_mode] == 'mask_inprogress':
      coords = self._set_rect_height_width(self.mask, self.cx, self.cy)
      self.canvas.restore_region(self.ax1_background)
      self.ax1.draw_artist(self.mask)
      self.fig.canvas.update()
      self.fig.canvas.flush_events()
    
  def _mask_re_press(self, event):
    if event.inaxes is None:
      return
    elif not str(event.inaxes.get_axes()) == str(self.ax1.get_axes()):	# is this self.ax1?
      return
    self.cx, self.cy = event.xdata, event.ydata				# get the cursor event data
    
    try:									# init/reset patch used for masking
      self.mask.remove()
    except (AttributeError, ValueError):
      pass 
    self.mask = self.ax1.add_patch(patches.Rectangle(				
      (0, 0),
      0,
      0,
      color='w',
      linestyle='dashed',
      fill=False)
    )
    self.ax1_background = self.canvas.copy_from_bbox(self.ax1.bbox)		# copy the axis background for blitting
    
    ox, oy = self._set_rect_origin(self.mask, self.cx, self.cy)

    self.ui.change_message_text("masking")   
    self._switch_current_mode('mask_inprogress')
    
  def _mask_re_release(self, event):
    self._switch_current_mode('mask_re')
    self.fig.canvas.draw()							# to clear
    if event.inaxes is None:
      return
    elif not str(event.inaxes.get_axes()) == str(self.ax1.get_axes()):	# is this self.ax1?
      return
    self.cx, self.cy = event.xdata, event.ydata
    if int(self.cx) == self.mask.get_x() or int(self.cy) == self.mask.get_y():
      self.ui.change_message_text("not a valid mask", critical=True)   
      return
    coords = self._set_rect_height_width(self.mask, self.cx, self.cy)
    self.ui.button_mask_param1.setText(str(coords[0][0]))
    self.ui.button_mask_param2.setText(str(coords[0][1]))
    self.ui.button_mask_param3.setText(str(coords[1][0]))
    self.ui.button_mask_param4.setText(str(coords[1][1]))
    self.ui.change_message_text("adjust/choose a value for mask and click apply")          

  def _plot(self, ax, x, y, line=None):
    ''' callable by _draw_profiles() draw [x] and [y] to [ax] '''
    if line is None:								# draw a new line if [line] is not populated
      ax.cla()
      ax.set_xlim([np.min(x), np.max(x)])
      line = ax.plot(x, y)
    else:									# else, just update [line] x/y data
      ax.set_xlim([np.min(x), np.max(x)])
      ax.set_ylim([np.min(y), np.max(y)])
      line[0].set_xdata(x)
      line[0].set_ydata(y)
    return line

  def _populate_ax1_data(self): 
    ''' change [ax1_data] to reflect current view/scaling '''
    x1, y1 = self.current_display_subsection[0]
    x2, y2 = self.current_display_subsection[1]
    if self.display_types[self.current_display] == 'magnitude':
      self.ax1_data = np.abs(self.data[y1:y2, x1:x2])**2
    elif self.display_types[self.current_display] == 'phase':
      self.ax1_data = np.angle(self.data[y1:y2, x1:x2])
    elif self.display_types[self.current_display] == 'real':
      self.ax1_data = np.real(self.data[y1:y2, x1:x2])
    elif self.display_types[self.current_display] == 'imag':
      self.ax1_data = np.imag(self.data[y1:y2, x1:x2])**2
    
    if self.data_scales[self.current_data_scaling] == 'linear':
      pass
    elif self.data_scales[self.current_data_scaling] == 'log':
      self.ax1_data[np.where(self.ax1_data == 0)] = 10**-10
      self.ax1_data = np.log10(self.ax1_data)
    elif self.data_scales[self.current_data_scaling] == '99.5%':
      upper_percentile = np.percentile(self.ax1_data, 99.5)
      lower_percentile = np.percentile(self.ax1_data, 0.5)
      self.ax1_data[np.where(self.ax1_data<lower_percentile)] = lower_percentile
      self.ax1_data[np.where(self.ax1_data>upper_percentile)] = upper_percentile
      
  def _resize_move(self, event):
    if event.inaxes is None:
      return
    elif not str(event.inaxes.get_axes()) == str(self.ax1.get_axes()):	# is this self.ax1?
      return
    self.cx, self.cy = event.xdata, event.ydata
    if self.modes[self.current_mode] == 'resize_inprogress':
      self._set_rect_height_width(self.rresize, self.cx, self.cy)
      self.canvas.restore_region(self.ax1_background)
      self.ax1.draw_artist(self.rresize)
      self.fig.canvas.update()
      self.fig.canvas.flush_events()

  def _resize_press(self, event):
    if event.inaxes is None:
      return
    elif not str(event.inaxes.get_axes()) == str(self.ax1.get_axes()):	# is this self.ax1?
      return
    self.cx, self.cy = event.xdata, event.ydata
    try:									# init/reset patch used for masking
      self.rresize.remove()
    except (AttributeError, ValueError):
      pass 
    self.rresize = self.ax1.add_patch(patches.Rectangle(
      (0, 0),
      0,
      0,
      color='w',
      fill=False)
    )
    self.ax1_background = self.canvas.copy_from_bbox(self.ax1.bbox)  
    
    self._set_rect_origin(self.rresize, self.cx, self.cy)
    
    self.ui.change_message_text("resizing")
    self._switch_current_mode('resize_inprogress')
    
  def _resize_release(self, event):
    self._switch_current_mode('resize')
    if event.inaxes is None:
      return
    elif not str(event.inaxes.get_axes()) == str(self.ax1.get_axes()):	# is this self.ax1?
      return
    self.cx, self.cy = event.xdata, event.ydata
    if int(self.cx) == self.rresize.get_x() or int(self.cy) == self.rresize.get_y():
      self.ui.change_message_text("not a valid resize box", critical=True)   
      return
    new_display_coordinates = self._set_rect_height_width(self.rresize, self.cx, self.cy)
    self.current_display_subsection = new_display_coordinates
    self._populate_ax1_data()
    self._draw_data()
    self._draw_xhair()
    self.fig.canvas.draw()
    self.ui.change_message_text("resized")
    
  def _set_circ_origin(self, circ, x, y):
    x = int(x)
    y = int(y)
    circ.center = (x, y)
    return x, y
    
  def _set_circ_radius(self, circ, x, y=None):
    if y==None:					# if explictly setting radius via text box
      r = int(x)
      circ.set_radius(r)
      return r
    else:
      ox, oy = circ.center
      width = int(np.abs(x-ox))
      height = int(np.abs(y-oy))
      r = int(np.sqrt((width**2) + (height**2)))
      circ.set_radius(r)
      return r
      
  def _set_rect_origin(self, rect, x, y):
    x = int(x)
    y = int(y)
    rect.set_xy((x, y))
    return x, y
    
  def _set_rect_height_width(self, rect, x, y):
    ox, oy = rect.get_xy()
    new_ox = int(np.min((ox, x)))
    new_oy = int(np.min((oy, y)))
    width = int(np.abs(x-ox))
    height = int(np.abs(y-oy))
    self._set_rect_origin(rect, new_ox, new_oy)
    rect.set_width(width)
    rect.set_height(height)
    
    x1 = int(new_ox)
    x2 = int(np.max((ox, x)))
    y1 = int(new_oy)
    y2 = int(np.max((oy, y)))
    
    return [(x1,y1), (x2,y2)]
  
  def _switch_current_mode(self, new_mode):
    ''' switch current mode '''
    if new_mode in self.modes:
      self.current_mode = self.modes.index(new_mode)
      return self.current_mode
    
  def connect_ui_to_events(self):
    self.ui.button_inspect.clicked.connect(self.connect_inspect_event_to_fig)
    self.ui.button_resize.clicked.connect(self.connect_resize_event_to_fig)
    self.ui.button_mask_re.clicked.connect(self.connect_mask_re_event_to_fig)
    self.ui.button_mask_ci.clicked.connect(self.connect_mask_ci_event_to_fig)
    self.ui.button_mask_apply.clicked.connect(self.connect_mask_apply_event_to_fig)
    self.ui.button_mask_param1.textChanged.connect(self.connect_mask_param_event_to_fig)
    self.ui.button_mask_param2.textChanged.connect(self.connect_mask_param_event_to_fig)
    self.ui.button_mask_param3.textChanged.connect(self.connect_mask_param_event_to_fig)
    self.ui.button_mask_param4.textChanged.connect(self.connect_mask_param_event_to_fig)
    self.ui.button_mask_value.textChanged.connect(self.connect_mask_value_event_to_fig)
    self.ui.button_fft.clicked.connect(self._action_fft)
    self.ui.button_fftshift.clicked.connect(self._action_fftshift)
    self.ui.button_nscaling.clicked.connect(self._control_scaling_next)
    self.ui.button_pscaling.clicked.connect(self._control_scaling_prev)
    self.ui.button_nview.clicked.connect(self._control_display_next)
    self.ui.button_pview.clicked.connect(self._control_display_prev)
    self.ui.button_resetview.clicked.connect(self._control_reset_view)
    self.ui.button_resetdata.clicked.connect(self._control_reset_data)
    if self.current_mode == '':
      pass

  def connect_inspect_event_to_fig(self):
    ''' input connects for inspect mode '''
    self._disconnect_all_connects()
    self.cidmousebuttonpress = self.fig.canvas.mpl_connect('button_press_event', self._inspect)
    self.ui.change_message_text("inspect mode (click)")
    self._switch_current_mode('inspect')
    
  def connect_mask_apply_event_to_fig(self):
    ''' input connects for applying mask '''
    if self.display_types[self.current_display] not in ['real', 'imag']:
      self.ui.change_message_text("can only alter real/imag components of data", critical=True)   
      return
    if self.modes[self.current_mode] == 'mask_re':
      x1, y1 = self.mask.get_xy()
      x2 = self.mask.get_width() + x1
      y2 = self.mask.get_height() + y1
      if x2 == x1 or y2 == y1:
	self.ui.change_message_text("mask invalid or not defined", critical=True)   
	return
      else:
	self.data[y1:y2, x1:x2] = float(self.ui.button_mask_value.text()) 
	self._populate_ax1_data()
	self._draw_data()
	self._draw_profiles()
	self._draw_xhair()
	self.fig.canvas.draw()
	self.ui.change_message_text("mask applied")
    elif self.modes[self.current_mode] == 'mask_ci':
      x1, y1 = self.mask.center
      r = self.mask.get_radius()
      if r==0:
	self.ui.change_message_text("mask invalid or not defined", critical=True)   
	return
      else:
	y, x = np.mgrid[0:self.data.shape[0], 0:self.data.shape[1]]
	y -= y1
	x -= x1  
	mask = x*x + y*y <= r*r
	self.data[mask] = float(self.ui.button_mask_value.text()) 
	self._populate_ax1_data()
	self._draw_data()
	self._draw_profiles()
	self._draw_xhair()
	self.fig.canvas.draw()
	self.ui.change_message_text("mask applied")
	  
  def connect_mask_ci_event_to_fig(self):
    ''' input connects for mask_ci mode '''
    self._disconnect_all_connects()
    self.cidmousebuttonpress = self.fig.canvas.mpl_connect('button_press_event', self._mask_ci_press)
    self.cidmousemove = self.fig.canvas.mpl_connect('motion_notify_event', self._mask_ci_move)
    self.cidmousebuttonrelease = self.fig.canvas.mpl_connect('button_release_event', self._mask_ci_release)
    self.ui.change_message_text("circular mask (drag)")
    self._switch_current_mode('mask_ci')
      
  def connect_mask_re_event_to_fig(self):
    ''' input connects for mask_re mode '''
    self._disconnect_all_connects()
    self.cidmousebuttonpress = self.fig.canvas.mpl_connect('button_press_event', self._mask_re_press)
    self.cidmousemove = self.fig.canvas.mpl_connect('motion_notify_event', self._mask_re_move)
    self.cidmousebuttonrelease = self.fig.canvas.mpl_connect('button_release_event', self._mask_re_release)
    self.ui.change_message_text("rectangular mask (drag)")
    self._switch_current_mode('mask_re')
    
  def connect_mask_param_event_to_fig(self):
    ''' input connects for mask mode, changing the value of params boxes '''
    try:
      val1 = int(self.ui.button_mask_param1.text())
    except (ValueError, UnboundLocalError):
      self.ui.change_message_text("param1 value is not an integer!", critical=True)
    try:
      val2 = int(self.ui.button_mask_param2.text())
    except (ValueError, UnboundLocalError):
      self.ui.change_message_text("param2 value is not an integer!", critical=True)
    try:
      val3 = int(self.ui.button_mask_param3.text())
    except (ValueError, UnboundLocalError):
      self.ui.change_message_text("param3 value is not an integer!", critical=True)
    try:
      val4 = int(self.ui.button_mask_param4.text())
    except (ValueError, UnboundLocalError):
      self.ui.change_message_text("param4 value is not an integer!", critical=True)	
    if self.modes[self.current_mode] == 'mask_re':
      try:
	val1, val2, val3, val4
	self._set_rect_origin(self.mask, val1, val2)
	self._set_rect_height_width(self.mask, val3, val4)
	self.fig.canvas.draw()	#FIXME
      except NameError:
	self.ui.change_message_text("mask params have not all been set!", critical=True)
    elif self.modes[self.current_mode] == 'mask_ci':
      try:
	val1, val2, val3
	self._set_circ_origin(self.mask, val1, val2)
	self._set_circ_radius(self.mask, val3)
	self.fig.canvas.draw()	#FIXME
      except NameError:
	self.ui.change_message_text("mask params have not all been set!", critical=True)     
    
  def connect_mask_value_event_to_fig(self):
    ''' input connects for mask mode, changing the value of text box '''
    try:
      val = float(self.ui.button_mask_value.text())
      self.ui.change_message_text("mask value changed (" + str(val) + ")")
    except ValueError:
      self.ui.change_message_text("mask value is not a number!", critical=True)

  def connect_resize_event_to_fig(self):
    ''' input connects for resize mode '''
    self._disconnect_all_connects()
    self.cidmousebuttonpress = self.fig.canvas.mpl_connect('button_press_event', self._resize_press)
    self.cidmousemove = self.fig.canvas.mpl_connect('motion_notify_event', self._resize_move)
    self.cidmousebuttonrelease = self.fig.canvas.mpl_connect('button_release_event', self._resize_release)
    self.ui.change_message_text("resize mode (drag)")
    self._switch_current_mode('resize') 

  def go(self):
    self.connect_inspect_event_to_fig()
    self.connect_ui_to_events()

    self.ui.main_frame.exec_()

    return self.data			# returns manipulated data
    
if __name__=="__main__":
  
  sampling = 256
  y, x = np.ogrid[-sampling/2:sampling/2, -sampling/2:sampling/2]
  mask = x*x + y*y <= ((sampling-1)/2)*((sampling-1)/2)
  data = np.zeros((sampling, sampling), dtype='complex')
  data[mask] = 1 + 0j
  data = np.pad(data, 256, mode='constant')
  
  app = QApplication(sys.argv)		# create app
  ui = ui()				# create ui instance
  e = editable(ui, data)		# create editable instance
  e.go()				# connect events and exec app
  
  
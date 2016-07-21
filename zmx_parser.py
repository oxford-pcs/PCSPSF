import codecs

import numpy as np
import pylab as plt

from util import sf

class zfftpsf():
  def __init__(self, fname, logger, verbose=True):
    self.fname = fname
    self.logger = logger
    self.header = {"WAVE": None, "FIELD": None, "WAVE_EXP": None, "DATA_SPACING": None, "DATA_SPACING_EXP": None, "DATA_AREA": None, "DATA_AREA_EXP": None, "PGRID_SIZE": None, "IGRID_SIZE": None, "CENTRE": None}
    self.data = None 
    self.verbose = verbose
    
  def _decode(self, encoding):
    fp = codecs.open(self.fname, "r", encoding)
    content = fp.readlines()
    fp.close()
    return content
  
  def parseFileHeader(self):
    content = self._decode("UTF-16-LE")
    for idx, line in enumerate(content):
      if idx == 8:
	self.header['WAVE'] = float(line.split()[0].strip())
	if unicode(line.split()[1].rstrip(',').strip()) == u'm':
	  self.header['WAVE_EXP'] = 1
	if unicode(line.split()[1].rstrip(',').strip()) == u'mm':
	  self.header['WAVE_EXP'] = 1e-3
	elif unicode(line.split()[1].rstrip(',').strip()) == u'\xb5m':
	  self.header['WAVE_EXP'] = 1e-6
	elif unicode(line.split()[1].rstrip(',').strip()) == u'nm':
	  self.header['WAVE_EXP'] = 1e-9
	self.header['FIELD'] = (float(line.split()[3].rstrip(',').strip()), float(line.split()[4].strip()))
      if idx == 9:
	self.header['DATA_SPACING'] = float(line.split()[3].strip())
	if unicode(line.split()[4].rstrip('.').strip()) == u'm':
	  self.header['DATA_SPACING_EXP'] = 1
	if unicode(line.split()[4].rstrip('.').strip()) == u'mm':
	  self.header['DATA_SPACING_EXP'] = 1e-3
	elif unicode(line.split()[4].rstrip('.').strip()) == u'\xb5m':
	  self.header['DATA_SPACING_EXP'] = 1e-6
	elif unicode(line.split()[4].rstrip('.').strip()) == u'nm':
	  self.header['DATA_SPACING_EXP'] = 1e-9
      if idx == 10:
	self.header['DATA_AREA'] = float(line.split()[3].strip())
	if unicode(line.split()[4].strip()) == u'm':
	  self.header['DATA_AREA_EXP'] = 1
	if unicode(line.split()[4].strip()) == u'mm':
	  self.header['DATA_AREA_EXP'] = 1e-3
	elif unicode(line.split()[4].strip()) == u'\xb5m':
	  self.header['DATA_AREA_EXP'] = 1e-6
	elif unicode(line.split()[4].strip()) == u'nm':
	  self.header['DATA_AREA_EXP'] = 1e-9
      if idx == 13:
	self.header['PGRID_SIZE'] = (int(line.split()[3].strip()), int(line.split()[5].strip()))
      if idx == 14:
	self.header['IGRID_SIZE'] = (int(line.split()[3].strip()), int(line.split()[5].strip()))
      if idx == 15:
	self.header['CENTRE'] = (int(line.split()[4].rstrip(',').strip()), int(line.split()[6].strip()))
      if None not in self.header.viewvalues():		# we've fully populated the header dict
	return True
    return False  
  
  def _parseFileData(self, sampling):
    content = self._decode("UTF-16-LE")
    data = []
    for idx, line in enumerate(content):
      try:
        if idx>=18:
	  data.append([float(i.rstrip('\r\n')) for i in line.split('\t')])  
      except TypeError:					# some non-floatable value has been found
	return False
    self.data = np.array(data)
    if not sampling == self.data.shape:			# not the same dimensions as expected sampling suggests
      return False
    return True 
  
  def parse(self, debug=False):
    if self.parseFileHeader():
      if debug:
        self.logger.debug(self.header)
      if self._parseFileData(self.header['IGRID_SIZE']):
        if debug:
          plt.imshow(self.data)
          plt.colorbar()
          plt.show()
        if self.verbose:
          self.logger.debug(" Successfully parsed ZEMAX file.")
      else:
	self.logger.critical(" Failed to read ZEMAX file data.")
    else:
      self.logger.critical(" Failed to read ZEMAX file header.")  
      
  def getHeader(self):
    return self.header 
  
  def getData(self, in_radians=False, match_pupil=None):
    return np.array(self.data)

class zwfe():
  def __init__(self, fname, logger, verbose=True):
    self.fname = fname
    self.logger = logger
    self.header = {"WAVE": None, "FIELD": None, "WAVE_EXP": None, "P2V": None, "RMS": None, "EXIT_PUPIL_DIAMETER": None, "SAMPLING": None, "CENTRE": None}
    self.data = None 
    self.verbose = verbose
    
  def _decode(self, encoding):
    fp = codecs.open(self.fname, "r", encoding)
    content = fp.readlines()
    fp.close()
    return content
  
  def parseFileHeader(self):
    content = self._decode("UTF-16-LE")
    for idx, line in enumerate(content):
      if idx == 8:
	self.header['WAVE'] = float(line.split()[0].strip())
	if unicode(line.split()[1].rstrip(',').strip()) == u'm':
	  self.header['WAVE_EXP'] = 1
	if unicode(line.split()[1].rstrip(',').strip()) == u'mm':
	  self.header['WAVE_EXP'] = 1e-3
	elif unicode(line.split()[1].rstrip(',').strip()) == u'\xb5m':
	  self.header['WAVE_EXP'] = 1e-6
	elif unicode(line.split()[1].rstrip(',').strip()) == u'nm':
	  self.header['WAVE_EXP'] = 1e-9
	self.header['FIELD'] = (float(line.split()[3].rstrip(',').strip()), float(line.split()[4].strip()))
      if idx == 9:
	self.header['P2V'] = float(line.split()[4].strip())
	self.header['RMS'] = float(line.split()[8].strip())
      if idx == 11:
	self.header['EXIT_PUPIL_DIAMETER'] = float(line.split()[3].strip())
	self.header['EXIT_PUPIL_DIAMETER_UNIT'] = str(line.split()[4].strip())
      if idx == 13:
	self.header['SAMPLING'] = (int(line.split()[3].strip()), int(line.split()[5].strip()))
      if idx == 14:
	self.header['CENTRE'] = (int(line.split()[4].rstrip(',').strip()), int(line.split()[6].strip()))
      if None not in self.header.viewvalues():		# we've fully populated the header dict
	return True
    return False  
  
  def _parseFileData(self, sampling):
    content = self._decode("UTF-16-LE")
    data = []
    for idx, line in enumerate(content):
      try:
        if idx>=16:
	  data.append([float(i.rstrip('\r\n')) for i in line.split('\t')])  
      except TypeError:					# some non-floatable value has been found
	return False
    self.data = np.array(data)
    if not sampling == self.data.shape:			# not the same dimensions as expected sampling suggests
      return False
    return True 
  
  def parse(self, debug=False):
    if self.parseFileHeader():
      if debug:
        self.logger.debug(self.header)
      if self._parseFileData(self.header['SAMPLING']):
        if debug:
          plt.imshow(self.data)
          plt.colorbar()
          plt.show()
        if self.verbose:
          self.logger.debug(" Successfully parsed ZEMAX file.")
      else:
	self.logger.critical(" Failed to read ZEMAX file data.")
    else:
      self.logger.critical(" Failed to read ZEMAX file header.")  
      
  def getHeader(self):
    return self.header 
  
  def getData(self, in_radians=False, match_pupil=None):
    '''
       If a pupil instance is given, the data will be fourier scaled.
    '''
    # translate unit to numerical quantity so output scale is physically meaningful
    if match_pupil is not None:
      if self.header['EXIT_PUPIL_DIAMETER_UNIT'] == "Millimeters":
	gsize_mfactor = 1e-3
      elif self.header['EXIT_PUPIL_DIAMETER_UNIT'] == "Meters":
	gsize_mfactor = 1
      else:
	gsize_mfactor = 1e-3
	self.logger.warning(" Unrecognised radius unit, assuming mm.")
	
      physical_exit_pupil_diameter = self.header['EXIT_PUPIL_DIAMETER']*gsize_mfactor
      
      wfe_plate_scale = physical_exit_pupil_diameter/self.data.shape[0]							# m/px	
      
      plate_scale_difference = wfe_plate_scale/(match_pupil.pupil_plate_scale*match_pupil.physical_gsize_mfactor)	# difference between plate scales
      adjusted_pupil_size = plate_scale_difference*self.data.shape[0]							# required size of reshaped array
      padding_required_post_fft = int(round((adjusted_pupil_size-self.data.shape[0])/2))				# padding required post fft (scale)
      
      if self.verbose:
	self.logger.debug(" Pupil plate scale is: " + sf((match_pupil.pupil_plate_scale*match_pupil.physical_gsize_mfactor*1e3), 2) + "mm/px")
	self.logger.debug(" WFE plate scale is: " + sf(wfe_plate_scale*1e3, 2) + "mm/px")
      rescaled_data = np.fft.fft2(self.data)
      rescaled_data = np.fft.fftshift(rescaled_data)    
      if padding_required_post_fft > 0:
        rescaled_data = np.pad(rescaled_data, padding_required_post_fft, mode='constant')	
      else:
	self.logger.debug(" Pupil plate scale is coarser than WFE plate scale, pick a larger pupil sampling!")
	exit(0)
      rescaled_data = np.fft.ifft2(rescaled_data)
      padding_required_post_ifft = (match_pupil.gsize-rescaled_data.shape[0])/2			# padding required post ifft (match entrance pupil array dimension)
      rescaled_data = np.pad(rescaled_data, padding_required_post_ifft, mode='constant')
      
      rescaled_data = rescaled_data*(np.max(np.abs(self.data))/np.max(np.abs(rescaled_data)))
      
      if in_radians:
        return np.abs(rescaled_data)*2*np.pi
      else:		
	return np.abs(rescaled_data)
    else:
      if in_radians:
        return np.abs(self.data)*2*np.pi
      else:
	return np.abs(self.data)*2*np.pi
    
	

	

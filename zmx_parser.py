import codecs
from decimal import *

import numpy as np
import pylab as plt

from util import sf, resample2d

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
  
  def getData(self):
    return np.array(self.data)
    
class zsystemdata():
  def __init__(self, fname, logger, verbose=True):
    self.fname = fname
    self.logger = logger
    self.keywords = {"WFNO": None, "EPD": None}
    self.verbose = verbose
    
  def _decode(self, encoding):
    fp = codecs.open(self.fname, "r", encoding)
    content = fp.readlines()
    fp.close()
    return content
  
  def _parseFileForKeywords(self):
    content = self._decode("UTF-16-LE")
    for idx, line in enumerate(content):
      if len(line.split(':')) >= 2:
	if "Working F/#" in line.split(':')[0]:
	  self.keywords['WFNO'] = float(line.split(':')[1].strip())
	elif "Entrance Pupil Diameter" in line.split(':')[0]:
	  self.keywords['EPD'] = float(line.split(':')[1].strip())
	if None not in self.keywords.viewvalues():		# we've fully populated the header dict
	  return True
    return False  
  
  def parse(self, debug=False):
    if self._parseFileForKeywords():
      if debug:
        self.logger.debug(self.keywords)
    else:
      self.logger.critical(" Failed to read ZEMAX keywords!")  
      exit(0)
      
  def getKeywords(self):
    return self.keywords
	
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
    try:
      for idx, line in enumerate(content):
	if idx == 8:
	  self.header['WAVE'] = Decimal(line.split()[0].strip())
	  if unicode(line.split()[1].rstrip(',').strip()) == u'm':
	    self.header['WAVE_EXP'] = Decimal('1')
	  if unicode(line.split()[1].rstrip(',').strip()) == u'mm':
	    self.header['WAVE_EXP'] = Decimal('1e-3')
	  elif unicode(line.split()[1].rstrip(',').strip()) == u'\xb5m':
	    self.header['WAVE_EXP'] = Decimal('1e-6')
	  elif unicode(line.split()[1].rstrip(',').strip()) == u'nm':
	    self.header['WAVE_EXP'] = Decimal('1e-9')
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
    except IndexError:
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
  
  def getData(self, EPD, match_pupil=None, in_radians=True):

    # Return WFE data, resampling and padding as appropriate.
    #
    data = self.data
    if match_pupil is not None:
      wfe_pupil_diameter 	= EPD									# mm
      wfe_plate_scale 		= wfe_pupil_diameter/data.shape[0]					# mm/px	
      self.logger.debug(" WFE map has a plate scale of " + str(sf(wfe_plate_scale, 2)) + "mm/px")
      
      # resample and crop to same plate scale as pupil
      if wfe_plate_scale*data.shape[0] < match_pupil.pupil_plate_scale*match_pupil.sampling:
	self.logger.critical(" WFE map extent is smaller than matched pupil extent, this would lead to extrapolation!")
	exit(0)
      wfe_s		 = -(data.shape[0]/2)*wfe_plate_scale
      wfe_e		 = -wfe_s
      match_pupil_s 	 = -(match_pupil.sampling/2)*match_pupil.pupil_plate_scale
      match_pupil_e 	 = -match_pupil_s
      data		 = resample2d(data, wfe_s, wfe_e, wfe_plate_scale, wfe_s, wfe_e, match_pupil.pupil_plate_scale, gauss_sig=0)
      
      self.logger.debug(" RMS wavefront error is " + str(sf(np.std(data), 2)) + " waves.")
      
      # pad the array to match pupil shape
      #
      # there's a caveat here, in that if the size of the pupil is odd, we will
      # be unable to pad the WFE array evenly. This equates to roughly a fraction 
      # of a percent misalignment in the worst case
      #
      pad_by = (match_pupil.gsize-data.shape[0])/2.
      if data.shape[0] % 2 != 0:
	pad_by = np.ceil(pad_by)				# adds 1 extra column and row of padding than required
	data = np.pad(data, int(pad_by), mode='constant')
	data = data[:-1,:-1]					# removes rightmost and bottommost padding
      else:
        data = np.pad(data, int(pad_by), mode='constant')
   
      # inverse fft shift to match pupil format
      data = np.fft.ifftshift(data)
    
      # convert to radians from waves, if requested
      if in_radians:
        data = np.abs(data)*2*np.pi
      else:
        data = np.abs(data)*2*np.pi   
    
      return data

	

import codecs
import numpy as np
import pylab as plt

class wf():
  def __init__(self, fname, logger):
    self.fname = fname
    self.logger = logger
    self.header = {"WAVE": None, "FIELD": None, "WAVE_EXP": None, "P2V": None, "RMS": None, "SAMPLING": None, "CENTRE": None}
    self.data = None
    
  def _decode(self, encoding):
    fp = codecs.open(self.fname, "r", encoding)
    content = fp.readlines()
    fp.close()
    return content
  
  def _parseFileHeader(self):
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
    if self._parseFileHeader():
      if debug:
        self.logger.debug(self.header)
      if self._parseFileData(self.header['SAMPLING']):
        if debug:
          plt.imshow(self.data)
          plt.colorbar()
          plt.show()
        self.logger.debug(" Successfully parsed ZEMAX file.")
      else:
	self.logger.critical(" Failed to read ZEMAX file data.")
    else:
      self.logger.critical(" Failed to read ZEMAX file header.")
      
  def getHeader(self):
    return self.header
   
  def getData(self):
    return self.data
    
	

	

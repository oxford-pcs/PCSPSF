class camera():
  '''
    Class to define spatial scales in the image plane given 
    a camera working f/#.
  '''
  def __init__(self, wfno):
    self.wfno = wfno
    self.s_resolution_element 	= None
    self.s_pscale		= None
    self.s_detector_FOV		= None
    self.s_airy_disk_d		= None
    
  def populateSpatialParameters(self, im):
    '''
      Take a conjugateImage class and calculates spatial parameters.
    '''
    self.s_resolution_element	= self.wfno*float(im.wave)*1e6			# micron/resolution element
    self.s_pscale		= self.s_resolution_element/im.pupil.gamma	# micron/px
    self.s_detector_FOV 	= self.s_pscale*im.pupil.gsize			# micron
    self.s_airy_disk_d		= 2.44*self.s_resolution_element		# micron
         
  def getDetectorHFOV(self):
    '''
      Returns the detector half field of view.
    '''
    return self.s_detector_FOV/2.
   
class pcamera():
  '''
    Class to define spatial scales in the image plane given a camera focal
    length [focal_length].
  '''
  def __init__(self, focal_length, pupil_diameter):
    self.wfno = focal_length / pupil_diameter
    
  def getLinearAiryDiskDiameter(self, wave):
    return 2.44*self.getLinearResolutionElement(wave)         # m
  
  def getLinearPixelScale(self, wave, pupil):
    return self.getLinearResolutionElement(wave)/pupil.gamma  # m/px
  
  def getLinearResolutionElement(self, wave):
    return self.wfno*float(wave)                              # m

  def getLinearDetectorFOV(self, wave, pupil):
    return self.getLinearPixelScale(wave, pupil)*pupil.gsize  # m

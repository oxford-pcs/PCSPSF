def sf(fig, n):
  format = '%.' + str(n) + 'g'
  return '%s' % float(format % float(fig))

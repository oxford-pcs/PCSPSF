# PCSPSF

## Overview

This package reconstructs the instrumental PSF after image-slicing. 

In brief, for each wavelength within the range specified:

1. A dummy telescope pupil is created
2. The conjugate image of this pupil is constructed and rescaled such that the plate scale matches that of a specified reference wavelength
3. The image is sliced up	
4. The image of each slice is moved into pupil space
5. The WFE from the collimator and/or camera are added by addition of the corresponding Zemax phase maps corresponding to the field position for each slice
6. The resulting aberrated pupils are moved back into image spaces
7. All slices are stitched together to make the composite image

A datacube can then be assembled, given a filename (`-f`) and output (`-o`).

## Dependencies

- Zemax
- PyZDDE
- zController
- instrument_builder

## Gotchas

The width of a single slice must be an integer number of pixels. This just makes it easier for everyone, namely me.

The program computes the angular pixel scale lambda / D assuming a pupil baffle D, such that the spatial scale gives a certain number of slices per resolution element. 

No consideration is currently given to anamorphic preoptics.

No consideration is currently given to the number of spectrographs (assumed one).



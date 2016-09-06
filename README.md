# psf-simulator

## Summary

This package can be used to analyse the resulting PSF following image slicing after phase errors have been 
added.

To summarise, the routine proceeds as, for each wavelength:

- Create a pupil
- Move to image plane and rescale image
- Slice up the image
- FFT back to pupil plane
- Add WFE
- FFT to image plane and composite slices

After which, a datacube is assembled and (optionally) resampled.

We must use an odd number of slices and the width of a single slice must be an integer number of pixels. This 
makes it easier computationally, and ensures that a single slice will lie on the centre of the PSF.


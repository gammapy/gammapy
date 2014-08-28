# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import gc
from ..stats import significance
from ..image import binary_dilation_circle


__all__ = ['GammaImages', 'IterativeKernelBackgroundEstimator']


class GammaImages(object):
    """TODO: implement a more general images container class
    that can be re-used in other places as well.
    """

    def __init__(self, counts, background=None, mask=None):
        self.counts = np.asarray(counts.data, dtype=float)
        self.header = counts.header
        if background == None:
            # Start with a flat background estimate
            self.background = np.ones_like(counts.data, dtype=float)
        else:
            self.background = np.asarray(background.data, dtype=float)

        if mask == None:
            self.mask = np.ones_like(counts.data, dtype=bool)
        else:
            self.mask = np.asarray(mask.data, dtype=bool)
    
    def compute_correlated_maps(self, kernel):
        """Compute significance image for a given kernel.
        """
        from scipy.ndimage import convolve

        self.counts_corr = convolve(self.counts, kernel)
        self.background_corr = convolve(self.background, kernel)
        self.significance = significance(self.counts_corr, self.background_corr)

        return self


class IterativeKernelBackgroundEstimator(object):
    """Iteratively estimate a background model.

    Parameters
    ----------
    images : `~gammapy.background.GammaImages`
        GammaImages object containing counts image and (optional) initial
        background estimation.
    source_kernel : `numpy.ndarray`
        Source kernel as a numpy array.
    background_kernel : `numpy.ndarray`
        Background convolution kernel as a numpy array.
    significance_threshold : float
        Significance threshold above which regions are excluded.
    mask_dilation_radius : float
        Amount by which mask is dilated with each iteration.
    delete_intermediate_results : bool
        Specify whether results of intermediate iterations should be deleted.
        (Otherwise, these are held in memory). Default True.
    save_intermediate_results : bool
        Specify whether to save intermediate results as FITS files to disk.
        Default False.
    filebase : str (optional)
        Base of filenames if save_intermediate_results = True. Default 'temp'.

    Returns
    -------
    mask : `~astropy.io.fits.ImageHDU`
        Final exclusion mask, when iterations are complete.
    background : `~astropy.io.fits.ImageHDU`
        Final background estimation, when iterations are complete.
    """

    def __init__(self, images, source_kernel, background_kernel,
                 significance_threshold, mask_dilation_radius,
                 delete_intermediate_results=True,
                 save_intermediate_results=False, filebase='temp'):
        
        # self._data[i] is a GammaImages object representing iteration number `i`.
        self._data = list()
        self._data.append(images)
        
        self.header = images.header
        
        self.source_kernel = source_kernel
        self.background_kernel = background_kernel

        self.significance_threshold = significance_threshold
        self.mask_dilation_radius = mask_dilation_radius
        
        self.delete_intermediate_results = delete_intermediate_results
        self.save_intermediate_results = save_intermediate_results

        # Calculate initial significance image
        self._data[-1].compute_correlated_maps(self.source_kernel)        
        gc.collect()
    
    def run_ntimes(self, n_iterations, filebase=None):
        """Run N iterations."""

        self._data[-1].compute_correlated_maps(self.source_kernel)
        if self.save_intermediate_results:
            self.save_files(filebase, index=0)

        for ii in range(1, n_iterations + 1):
            if ii == 1:
                # This is needed to avoid excluding the whole Galactic plane
                # in case the initial background estimate is much too low.
                update_mask = False
            else:
                update_mask = True
            
            self.run_iteration(update_mask)
            
            if self.save_intermediate_results: 
                self.save_files(filebase, index=ii)

            if self.delete_intermediate_results:
                # Remove results from previous iteration
                del self._data[0]
                gc.collect()

            mask = self.mask_image_hdu
            background = self.background_image_hdu

        return mask, background


    def run(self, filebase=None):
        """Run until mask is stable."""

        self._data[-1].compute_correlated_maps(self.source_kernel)
        if self.save_intermediate_results:
            self.save_files(filebase, index=0)
        
        self.run_iteration(update_mask=False)
            
        if self.save_intermediate_results: 
            self.save_files(filebase, index=ii)
        
        check_mask = False

        while check_mask == False:
            mask = self.mask_image_hdu
            old_mask = mask.data
            self.run_iteration(update_mask=True)
            
            if self.save_intermediate_results: 
                self.save_files(filebase, index=ii)

            if self.delete_intermediate_results:
                # Remove results from previous iteration
                del self._data[0]
                gc.collect()

            new_mask = self.mask_image_hdu
            background = self.background_image_hdu
            
            # Test mask (note will terminate as soon as mask does not change).
            new = np.nan_to_num(new_mask.data.sum())
            old = np.nan_to_num(old_mask.sum())

            if new == old:
                check_mask = True
            else:
                check_mask = False

        return new_mask, background


    def run_iteration(self, update_mask=True):
        """Run one iteration.

        Parameters
        ----------
        update_mask : bool
            Specify whether to update the exclusion mask stored in the input
            `~gammapy.background.GammaImages` object with the exclusion mask
            newly calculated in this method.
        """

        from scipy.ndimage import convolve
        # Start with images from the last iteration
        images = self._data[-1]

        # Compute new exclusion mask:
        if update_mask:
            mask = np.where(images.significance > self.significance_threshold, 0, 1)
            mask = np.invert(binary_dilation_circle(mask == 0, radius=self.mask_dilation_radius))
        else:
            mask = images.mask.copy()
        
        # Compute new background estimate:
        # Convolve old background estimate with background kernel,
        # excluding sources via the old mask.
        weighted_counts = convolve(images.mask * images.counts, self.background_kernel)
        weighted_counts_normalisation = convolve(images.mask.astype(float), self.background_kernel)
        background = weighted_counts / weighted_counts_normalisation
        
        # Convert new Images to HDUs to store in a GammaImages object
        counts = fits.ImageHDU(data=images.counts, header=images.header)
        background = fits.ImageHDU(data=background, header=images.header)
        mask = fits.ImageHDU(data=mask.astype(int), header=images.header)
        images = GammaImages(counts, background, mask)
        images.compute_correlated_maps(self.source_kernel)
        significance = fits.ImageHDU(data=images.significance, header=images.header)

        self._data.append(images)
    
    def save_files(self, filebase, index):
        """Saves files to fits."""

        header = self.header
        images = self._data[-1]

        filename = filebase + '{0:02d}_mask'.format(index) + '.fits'
        hdu = fits.ImageHDU(data=images.mask.astype(np.uint8), header=header)
        hdu.writeto(filename, clobber=True)

        filename = filebase + '{0:02d}_background'.format(index) + '.fits'
        hdu = fits.ImageHDU(data=images.background, header=header)
        hdu.writeto(filename, clobber=True)

        filename = filebase + '{0:02d}_significance'.format(index) + '.fits'
        hdu = fits.ImageHDU(data=images.significance, header=header)
        hdu.writeto(filename, clobber=True)    

    @property
    def mask_image_hdu(self):
        """Returns mask as `~astropy.io.fits.ImageHDU`."""

        header = self.header
        images = self._data[-1]

        return fits.ImageHDU(data=images.mask.astype(np.uint8), header=header)    

    @property
    def background_image_hdu(self):
        """Returns resulting background estimate as `~astropy.io.fits.ImageHDU`."""

        header = self.header
        images = self._data[-1]

        return fits.ImageHDU(data=images.background, header=header)

    @property
    def significance_image_hdu(self):
        """Returns resulting background estimate as `~astropy.io.fits.ImageHDU`."""

        header = self.header
        images = self._data[-1]

        return fits.ImageHDU(data=images.significance, header=header)

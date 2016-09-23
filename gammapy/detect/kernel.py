# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.io import fits
from astropy.convolution import Tophat2DKernel
from ..extern.pathlib import Path
from ..stats import significance
from ..image import SkyMask

__all__ = [
    'KernelBackgroundEstimatorData',
    'KernelBackgroundEstimator',
]


class KernelBackgroundEstimatorData(object):
    """TODO: implement a more general images container class
    that can be re-used in other places as well.
    """

    def __init__(self, counts, background=None, mask=None, header=None):
        self.counts = counts

        if header:
            self.header = header

        if background is not None:
            self.background = background

        if mask is None:
            self.mask = np.ones_like(counts, dtype=bool)
        else:
            self.mask = np.asarray(mask, dtype=bool)

    def initial_background(self, kernel):
        """Computes initial background estimation
        """
        from scipy.ndimage import convolve
        self.background = convolve(self.counts, kernel)
        return self

    def compute_correlated_images(self, kernel):
        """Compute significance image for a given kernel.
        """
        from scipy.ndimage import convolve

        if self.background is None:
            self.initial_background(kernel)

        self.counts_corr = convolve(self.counts, kernel)
        self.background_corr = convolve(self.background, kernel)
        self.significance = np.nan_to_num(significance(self.counts_corr,
                                                       self.background_corr))
        return self


class KernelBackgroundEstimator(object):
    """Estimate background using a source and background kernel.

    Output is a background image and exclusion mask, which can be used
    to other images, e.g. an excess, flux or significance image.

    Parameters
    ----------
    images : `KernelBackgroundEstimatorData`
        Counts image and (optional) initial background estimate.
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
    base_dir : str (optional)
        Base of filenames if save_intermediate_results = True. Default 'temp'.


    See Also
    --------
    gammapy.background.RingBackgroundEstimator,
    gammapy.background.AdaptiveRingBackgroundEstimator
    """

    def __init__(self, images, source_kernel, background_kernel,
                 significance_threshold, mask_dilation_radius,
                 delete_intermediate_results=True,
                 save_intermediate_results=False, base_dir='temp'):

        self.source_kernel = source_kernel
        self.background_kernel = background_kernel

        images.initial_background(background_kernel)
        # self._data[i] is a GammaImages object representing iteration number `i`.
        self._data = list()
        self._data.append(images)

        self.header = images.header

        self.significance_threshold = significance_threshold
        self.mask_dilation_radius = mask_dilation_radius

        self.delete_intermediate_results = delete_intermediate_results
        self.save_intermediate_results = save_intermediate_results
        # Calculate initial significance image
        self._data[-1].compute_correlated_images(self.source_kernel)

    def run(self, base_dir=None, max_iterations=10):
        """Run iterations until mask does not change (stopping condition).

        Parameters
        ----------
        base_dir : str
            Base string for filenames if iterations are saved to disk.
            Default None.
        max_iterations : int
            Maximum number of iterations after which the algorithm is
            terminated, if the termination condition (no change of mask between
            iterations) is not already satisfied.

        Returns
        -------
        mask : array-like
            Boolean array for the final mask after iterations are ended.
        background : array-like
            Array of floats for the final background estimation after
            iterations are complete.
        """

        if self.save_intermediate_results:
            self.save_files(base_dir, index=0)

        for ii in np.arange(max_iterations):
            self.run_iteration()

            if self.save_intermediate_results:
                self.save_files(base_dir, index=ii)

            # Dilate old mask to compare with new mask
            old_mask = self._data[0].mask
            new_mask = self._data[-1].mask

            if ii >= 3:
                # Prevents early termination in first two steps when
                # mask is not yet correct
                if np.alltrue(old_mask == new_mask):
                    continue

            if self.delete_intermediate_results:
                # Remove results from previous iteration
                del self._data[0]

        mask = self._data[-1].mask.astype(np.uint8)
        background = self._data[-1].background

        return mask, background

    def run_iteration(self, update_mask=True):
        """Run one iteration.

        Parameters
        ----------
        update_mask : bool
            Specify whether to update the exclusion mask stored in the input
            data object with the exclusion mask
            newly calculated in this method.
        """
        from scipy.ndimage import convolve
        # Start with images from the last iteration. If not, makes one.
        # Check if initial mask exists:
        if len(self._data) >= 2:
            mask = np.where(self._data[-1].significance > self.significance_threshold, 0, 1)
        else:
            # Compute new exclusion mask:
            if update_mask:
                mask = self._data[-1].significance > self.significance_threshold
                structure = Tophat2DKernel(self.mask_dilation_radius)
                mask = SkyMask(data=mask).dilate(structure)
                mask = np.invert(mask.data)
            else:
                mask = self._data[-1].mask.copy()

        # Compute new background estimate:
        # Convolve old background estimate with background kernel,
        # excluding sources via the old mask.
        weighted_counts = convolve(mask * self._data[-1].counts,
                                   self.background_kernel)
        weighted_counts_normalisation = convolve(mask.astype(float),
                                                 self.background_kernel)

        background = weighted_counts / weighted_counts_normalisation
        counts = self._data[-1].counts
        mask = mask.astype(int)

        images = KernelBackgroundEstimatorData(counts, background, mask)
        images.compute_correlated_images(self.source_kernel)
        self._data.append(images)

    def save_files(self, base_dir, index):
        """Saves files to fits."""
        base_dir = Path(base_dir)

        header = self.header

        filename = base_dir / '{0:02d}_mask.fits'.format(index)
        hdu = fits.ImageHDU(data=self._data[-1].mask.astype(np.uint8),
                            header=header)
        hdu.writeto(str(filename), clobber=True)

        filename = base_dir / '{0:02d}_background.fits'.format(index)
        hdu = fits.ImageHDU(data=self._data[-1].background, header=header)
        hdu.writeto(str(filename), clobber=True)

        filename = base_dir / '{0:02d}_significance.fits'.format(index)
        hdu = fits.ImageHDU(data=self._data[-1].significance, header=header)
        hdu.writeto(str(filename), clobber=True)

    @property
    def mask_image_hdu(self):
        """Mask (`~astropy.io.fits.ImageHDU`)"""
        return fits.ImageHDU(data=self._data[-1].mask.astype(np.uint8),
                             header=self.header)

    @property
    def background_image_hdu(self):
        """Background estimate (`~astropy.io.fits.ImageHDU`)"""
        return fits.ImageHDU(data=self._data[-1].background,
                             header=self.header)

    @property
    def significance_image_hdu(self):
        """Significance estimate (`~astropy.io.fits.ImageHDU`)"""
        return fits.ImageHDU(data=self._data[-1].significance,
                             header=self.header)

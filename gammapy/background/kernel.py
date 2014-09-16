# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import logging
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS

__all__ = ['GammaImages', 'IterativeKernelBackgroundEstimator']


class GammaImages(object):
    """Container for a set of related images.
    
    Meaning of mask:
    * 1 = background region
    * 0 = source region
    (such that multiplying with the mask zeros out the source regions)

    TODO: document
    """
    def __init__(self, counts, background=None, mask=None):
        self.counts = np.asarray(counts, dtype=float)

        if background == None:
            # Start with a flat background estimate
            self.background = np.ones_like(background, dtype=float)
        else:
            self.background = np.asarray(background, dtype=float)

        if mask == None:
            self.mask = np.ones_like(counts, dtype=bool)
        else:
            self.mask = np.asarray(mask, dtype=bool)
    
    def compute_correlated_maps(self, kernel):
        """Compute significance image for a given kernel.
        """
        #import IPython; IPython.embed()
        self.counts_corr = convolve(self.counts, kernel)
        self.background_corr = convolve(self.background, kernel)
        self.significance = significance(self.counts_corr, self.background_corr)

        return self

    def print_info(self):
        logging.info('Counts sum: {0}'.format(self.counts.sum()))
        logging.info('Background sum: {0}'.format(self.background.sum()))
        background_fraction = 100. * self.background.sum() / self.counts.sum()
        logging.info('Background fraction: {0}'.format(background_fraction))
        excluded_fraction = 100. * (1 - np.mean(self.mask))
        logging.info('Excluded fraction: {0}%'.format(excluded_fraction))
    
    def save(self, filename):
        logging.info('Writing {0}'.format(filename))


class IterativeBackgroundEstimator(object):
    """Iteratively estimate a background model.

    see: SciNeGHE source_diffuse_estimation.py
    TODO: document

    Parameters
    ----------
    image : `GammaImages`
        Gamma images

    See also
    --------
    `gammapy.detect.CWT`
    """
    def __init__(self, images, source_kernel, background_kernel,
                 significance_threshold, mask_dilation_radius,
                 delete_intermediate_results=True):
        
        # self._data[i] is a GammaImages object representing iteration number `i`.
        self._data = list()
        self._data.append(images)
        
        self.source_kernel = source_kernel
        self.background_kernel = background_kernel

        self.significance_threshold = significance_threshold
        self.mask_dilation_radius = mask_dilation_radius
        
        self.delete_intermediate_results = delete_intermediate_results
        
        gc.collect()
    
    def run(self, n_iterations, filebase):
        """Run N iterations."""

        self._data[-1].compute_correlated_maps(self.source_kernel)
        self.save_files(filebase, index=0)

        for ii in range(1, n_iterations + 1):
            logging.info('Running iteration #{0}'.format(ii))
            if ii == 1:
                # This is needed to avoid excluding the whole Galactic plane
                # in case the initial background estimate is much too low.
                update_mask = False
            else:
                update_mask = True
            
            self.run_iteration(update_mask)

            self.save_files(filebase, index=ii)

            if self.delete_intermediate_results:
                # Remove results from previous iteration
                del self._data[0]
                gc.collect()

    def run_iteration(self, update_mask=True):
        """Run one iteration."""
        # Start with images from the last iteration
        images = self._data[-1]
        
        logging.info('*** INPUT IMAGES ***')
        images.print_info()

        # Compute new exclusion mask:
        if update_mask:
            logging.info('Computing new exclusion mask')
            mask = np.where(images.significance > self.significance_threshold, 0, 1)
            #print('===', (mask == 0).sum())
            mask = np.invert(binary_dilation_circle(mask == 0, radius=self.mask_dilation_radius))
            #print('===', (mask == 0).sum())
        else:
            mask = images.mask.copy()
        
        # Compute new background estimate:
        # Convolve old background estimate with background kernel,
        # excluding sources via the old mask.
        weighted_counts = convolve(images.mask * images.counts, self.background_kernel)
        weighted_counts_normalisation = convolve(images.mask.astype(float), self.background_kernel)
        background = weighted_counts / weighted_counts_normalisation
        
        # Store new images
        images = GammaImages(counts, background, mask)
        logging.info('Computing source kernel correlated images.')
        images.compute_correlated_maps(self.source_kernel)

        logging.info('*** OUTPUT IMAGES ***')
        images.print_info()
        self._data.append(images)
    
    def save_files(self, filebase, index):

        # TODO: header should be stored as class member instead
        # This is a hack:
        header = fits.getheader('sources.fits.gz', 1)

        images = self._data[-1]

        filename = filebase + '{0:02d}_mask'.format(index) + '.fits'
        logging.info('Writing {0}'.format(filename))
        hdu = fits.ImageHDU(data=images.mask.astype(np.uint8), header=header)
        hdu.writeto(filename, clobber=True)

        filename = filebase + '{0:02d}_background'.format(index) + '.fits'
        logging.info('Writing {0}'.format(filename))
        hdu = fits.ImageHDU(data=images.background, header=header)
        hdu.writeto(filename, clobber=True)

        filename = filebase + '{0:02d}_significance'.format(index) + '.fits'
        logging.info('Writing {0}'.format(filename))
        hdu = fits.ImageHDU(data=images.significance, header=header)
        hdu.writeto(filename, clobber=True)

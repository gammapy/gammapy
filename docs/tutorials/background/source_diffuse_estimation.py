"""Estimate a diffuse emission model from Fermi LAT data.
"""
import gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from gammapy.image.utils import binary_dilation_circle, binary_disk

# Parameters
TOTAL_COUNTS = 1e6
SOURCE_FRACTION = 0.2

CORRELATION_RADIUS = 3
SIGNIFICANCE_THRESHOLD = 3
MASK_DILATION_RADIUS = 3 # pix
NUMBER_OF_ITERATIONS = 4

# Derived parameters
DIFFUSE_FRACTION = 1. - SOURCE_FRACTION

# Load example model images
source_image_true = fits.getdata('sources.fits.gz')
diffuse_image_true = fits.getdata('diffuse.fits.gz')

# Generate example data
source_image_true *= SOURCE_FRACTION * TOTAL_COUNTS / source_image_true.sum()
diffuse_image_true *= DIFFUSE_FRACTION * TOTAL_COUNTS / diffuse_image_true.sum()
total_image_true = source_image_true + diffuse_image_true

counts = np.random.poisson(total_image_true)

print('source counts: {0}'.format(source_image_true.sum()))
print('diffuse counts: {0}'.format(diffuse_image_true.sum()))

# If you want to check the input images plot them here:
#plt.figure(figsize=(0,10))
#plt.imshow(source_image_true)
#plt.imshow(np.log(counts))

import logging
logging.basicConfig(level=logging.INFO)
from scipy.ndimage import convolve
from gammapy.stats import significance

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
            

if __name__ == '__main__':
    # Start with flat background estimate
    background=np.ones_like(counts, dtype=float)
    images = GammaImages(counts=counts, background=background)

    #source_kernel = np.ones((5, 5))
    source_kernel = binary_disk(CORRELATION_RADIUS).astype(float)
    source_kernel /= source_kernel.sum()

    background_kernel = np.ones((10, 100))

    ibe = IterativeBackgroundEstimator(
                                       images=images,
                                       source_kernel=source_kernel,
                                       background_kernel=background_kernel,
                                       significance_threshold=SIGNIFICANCE_THRESHOLD,
                                       mask_dilation_radius=MASK_DILATION_RADIUS
                                       )

    ibe.run(n_iterations=4, filebase='test')

    ibe.run_iteration()
    #import IPython; IPython.embed()
    #ibe.save('test')
    
    #counts_hdu = background_hdu = mask_hdu = fits.open('sources.fits.gz')[1]
    #counts_hdu.data = images.counts
    #counts_hdu.writeto('testcounts.fits', clobber=True)
    #background_hdu.data = images.counts
    #background_hdu.writeto('testbackground.fits', clobber=True)
    #mask_hdu.data = images.mask.astype(int)
    #mask_hdu.writeto('testmask.fits', clobber=True)

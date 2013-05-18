"""Functions to compute TS maps

This is in the exploratory phase, we are trying to get a fast tool for a large map.
Here we compare different ways to split the map into parts and different optimizers.

Reference : Stewart (2009) "Maximum-likelihood detection of sources among Poissonian noise"
            Appendix A: Cash amplitude fitting
            http://adsabs.harvard.edu/abs/2009A%26A...495..989S

TODO:
- try different optimizers
- give good fit start values
- PSF-convolved Gauss-source kernels
- Use multiple CPUs with multiprocessing.
- check that Li & Ma significance maps match sqrt(ts) maps for kernel with weights 0 / 1 only
- implement On / Off and On Likelihood fitting
- implement optimized linear filter from Stewart paper
- implement down-sampling for large kernels or generally for speed
- implement possibility to only compute part of the TS image
- understand negative amplitudes!???
- speed profiling
  - expect speed constant with image size
  - expect speed inversely proportional to number of pixels in the kernel
  - expect speedup proportional to number of cores
- accuracy profiling:
  - want accuracy of TS = 0.1 for all regimes; no need to waste cycles on higher accuracy
  - don't care about accuracy for TS < 1
- check distribution against expected chi2(ndf) distribution
- HGPS survey sensitiviy calculation (maybe needs cluster computing?)
"""
import numpy as np
from .. import stats

__all__ = ['compute_ts_image', 'TSMapCalculator']

def fit_amplitude(counts, background, kernel, start_value):
    out = dict()
    def stat(amplitude):
        return stats.cash(counts, background + amplitude * kernel)
    
    from iminuit import Minuit
    minuit = Minuit(stat, pedantic=False, print_level=0,
                    amplitude=start_value)
    minuit.migrad()
    #import IPython; IPython.embed(); 1/0
    out['amplitude'] = minuit.values['amplitude']
    out['ncalls'] = minuit.ncalls
    return out

def compute_ts(images, kernel):
    # For the kernel we have to make a copy, otherwise
    # we modify the kernel in-place and will get incorrect
    # results for the next pixel 
    normalized_kernel = kernel /  kernel.sum()
    counts = images['counts']
    background = images['background']
    C0 = stats.cash(counts, background)
    out = fit_amplitude(counts, background, normalized_kernel)
    C1 = stats.cash(counts, background + out['amplitude'] * normalized_kernel)
    # Cash is a negative log likelihood statistic,
    # thus the minus in the TS formula here
    out['ts'] = - 2 * (C1 - C0)
    return out

def process_image_full(images, kernel, out, process_image_part):
    """
    images : dict with values as numpy arrays
    kernel : PSF-convolved source model
             kernel shape must be odd-valued
    out : dict of numpy arrays to fill
    process_image_part : function to process a part of the images
    
    TODO: Add different options to treat the edges!
    """
    n0, n1 = out.values()[0].shape

    # Check kernel shape
    k0, k1 = kernel.shape
    if (k0 % 2 == 0) or (k1 % 2 == 0):
        raise ValueError('Kernel shape must have odd dimensions')
    k0, k1 = k0 / 2, k1 / 2

    # Loop over all pixels
    for i0 in range(0, n0):
        for i1 in range(0, n1):
            image_parts = dict()
            # Cut out relevant parts of the image arrays
            # This creates views, i.e. is fast and memory efficient
            for name, image in images.items():
                i0_lo = min(k0, i0)
                i1_lo = min(k1, i1)
                i0_up = min(k0, n0 - i0 - 1)
                i1_up = min(k1, n1 - i1 - 1)
                part = image[i0 - i0_lo : i0 + i0_up,
                             i1 - i1_lo : i1 + i1_up]
                image_parts[name] = part
            # Cut out relevant part of the kernel array
            # This only applies when close to the edge
            kernel_part = kernel[k0 - i0_lo : k0 + i0_up,
                                 k1 - i1_lo : k1 + i1_up]

            out_part = process_image_part(image_parts, kernel_part)

            for name, image in out.items():
                out[name][i0, i1] = out_part[name]

    return out

def compute_ts_image(images, kernel):
    out = dict()
    out['ts'] = np.zeros_like(images['counts'], dtype='float64')
    out['ncalls'] = np.zeros_like(images['counts'], dtype='uint16')
    return process_image_full(images, kernel, out, compute_ts)



class TSMapCalculator(object):
    """TS Map calculator class."""
    
    def __init__(self, images, kernel, optimizer='migrad', guess_method='estimate'):
        self.images = images

        # Check kernel shape
        k0, k1 = kernel.shape
        if (k0 % 2 == 0) or (k1 % 2 == 0):
            raise ValueError('Kernel shape must have odd dimensions')
        self.kernel = kernel

        self.optimizer = optimizer
        self.guess_method = guess_method
        self.out_shape = self.images['counts'].shape

    def run(self):
        out = dict()
        out['ts'] = np.zeros_like(self.out_shape, dtype='float64')
        out['ncalls'] = np.zeros_like(self.out_shape, dtype='uint16')

        """
        # TODO: finish implementation 
        method =  ts_method()
        process_image_full(images, kernel, out, compute_ts)
        self.out = out
        """

    def _process_all_pixels(self):
        """Process all pixels"""
        n0, n1 = self.out_shape
        kernel = self.kernel
        k0, k1 = kernel.shape[0] / 2, kernel.shape[1] / 2
    
        # Loop over all pixels
        for i0 in range(0, n0):
            for i1 in range(0, n1):
                image_parts = dict()
                # Cut out relevant parts of the image arrays
                # This creates views, i.e. is fast and memory efficient
                for name, image in self.images.items():
                    i0_lo = min(k0, i0)
                    i1_lo = min(k1, i1)
                    i0_up = min(k0, n0 - i0 - 1)
                    i1_up = min(k1, n1 - i1 - 1)
                    part = image[i0 - i0_lo : i0 + i0_up,
                                 i1 - i1_lo : i1 + i1_up]
                    image_parts[name] = part
                # Cut out relevant part of the kernel array
                # This only applies when close to the edge
                kernel_part = kernel[k0 - i0_lo : k0 + i0_up,
                                     k1 - i1_lo : k1 + i1_up]

                self._process_one_pixel(image_parts, kernel_part)

    def _process_one_pixel(self):
        """Process one pixel"""
        # TODO: finish implementation
        """
                for name, image in self.out.items():
                    self.out[name][i0, i1] = out_part[name]
        """
        return out

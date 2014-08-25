# Licensed under a 3-clause BSD style license - see LICENSE.rst
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
- speed profiling:

  - expect speed constant with image size
  - expect speed inversely proportional to number of pixels in the kernel
  - expect speedup proportional to number of cores

- accuracy profiling:

  - want accuracy of TS = 0.1 for all regimes; no need to waste cycles on higher accuracy
  - don't care about accuracy for TS < 1

- check distribution against expected chi2(ndf) distribution
- HGPS survey sensitiviy calculation (maybe needs cluster computing?)
"""
from __future__ import print_function, division
import numpy as np
from .. import stats
from ..image import process_image_pixels

__all__ = ['ts_image',
           'TSMapCalculator',
           ]


def fit_amplitude(counts, background, kernel, start_value):
    """Fit amplitude.

    TODO: document.
    """
    out = dict()

    def stat(amplitude):
        return stats.cash(counts, background + amplitude * kernel)

    from iminuit import Minuit
    minuit = Minuit(stat, pedantic=False, print_level=0,
                    amplitude=start_value)
    minuit.migrad()
    # import IPython; IPython.embed(); 1/0
    out['amplitude'] = minuit.values['amplitude']
    out['ncalls'] = minuit.ncalls
    return out


def ts_center(images, kernel):
    """Compute TS for one position.

    The shapes of the images and the kernel must match.

    TODO: document
    """
    counts = np.asanyarray(images['counts'])
    background = np.asanyarray(images['background'])
    kernel = kernel / kernel.sum()

    assert counts.shape == kernel.shape
    assert background.shape == kernel.shape

    C0 = stats.cash(counts, background)
    out = fit_amplitude(counts, background, kernel)
    C1 = stats.cash(counts, background + out['amplitude'] * kernel)
    # Cash is a negative log likelihood statistic,
    # thus the minus in the TS formula here
    out['ts'] = - 2 * (C1 - C0)
    return out


def ts_image(images, kernel, extra_info=False):
    """Compute TS image.

    TODO: document
    """
    out = dict()
    out['ts'] = np.zeros_like(images['counts'], dtype='float64')
    out['ncalls'] = np.zeros_like(images['counts'], dtype='uint16')
    process_image_pixels(images, kernel, out, ts_center)
    if extra_info:
        return out
    else:
        return out['ts']


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

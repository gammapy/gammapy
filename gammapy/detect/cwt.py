# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import logging
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

__all__ = ['CWT']


def gauss_kernel(radius, n_sigmas=8):
    """Normalized 2D gauss kernel array.
    """
    sizex = int(n_sigmas * radius)
    sizey = int(n_sigmas * radius)
    radius = float(radius)
    xc = 0.5 * sizex
    yc = 0.5 * sizey
    y, x = np.mgrid[0:sizey - 1, 0:sizex - 1]
    x = x - xc
    y = y - yc
    x = x / radius
    y = y / radius
    g = np.exp(-0.5 * (x ** 2 + y ** 2))
    return g / (2 * np.pi * radius ** 2)  # g.sum()


def difference_of_gauss_kernel(radius, scale_step, n_sigmas=8):
    """Difference of 2 Gaussians (i.e. Mexican hat) kernel array.
    """
    sizex = int(n_sigmas * scale_step * radius)
    sizey = int(n_sigmas * scale_step * radius)
    radius = float(radius)
    xc = 0.5 * sizex
    yc = 0.5 * sizey
    y, x = np.mgrid[0:sizey - 1, 0:sizex - 1]
    x = x - xc
    y = y - yc
    x1 = x / radius
    y1 = y / radius
    g1 = np.exp(-0.5 * (x1 ** 2 + y1 ** 2))
    g1 = g1 / (2 * np.pi * radius ** 2)  # g1.sum()
    x1 = x1 / scale_step
    y1 = y1 / scale_step
    g2 = np.exp(-0.5 * (x1 ** 2 + y1 ** 2))
    g2 = g2 / (2 * np.pi * radius ** 2 * scale_step ** 2)  # g2.sum()
    return g1 - g2


class CWT(object):
    """Continuous wavelet transform.

    TODO: describe algorithm

    TODO: give references

    Initialization of wavelet family.

    Parameters
    ----------
    min_scale : float
        first scale used
    nscales : int
        number of scales considered
    scale_step : float
        base scaling factor
    """
    def __init__(self, min_scale, nscales, scale_step):
        self.kernbase = dict()
        self.scales = dict()
        self.nscale = nscales
        self.scale_step = scale_step
        for ns in np.arange(0, nscales):
            scale = min_scale * scale_step ** ns
            self.scales[ns] = scale
            self.kernbase[ns] = difference_of_gauss_kernel(scale, scale_step)

        # TODO: do we need self.scales and self.scale_array?
        self.scale_array = (scale_step ** (np.arange(0, nscales))) * min_scale

        max_scale = min_scale * scale_step ** nscales
        self.kern_approx = gauss_kernel(max_scale)

#        self.transform = dict()
#        self.error = dict()
#        self.support = dict()

        self.header = None
        self.wcs = None

    def set_data(self, image, background):
        """Set input images."""
        # TODO: check that image and background are consistent 
        self.image = image - 0.0
        self.nx, self.ny = self.image.shape
        self.filter = np.zeros((self.nx, self.ny)) 
        self.background = background - 0.0  # hack because of some bug with old version of fft in numpy
        self.model = np.zeros((self.nx, self.ny)) 
        self.approx = np.zeros((self.nx, self.ny))

        self.transform = np.zeros((self.nscale, self.nx, self.ny))
        self.error = np.zeros((self.nscale, self.nx, self.ny)) 
        self.support = np.zeros((self.nscale, self.nx, self.ny))

    def set_file(self, filename):
        """Set input images from FITS file"""
        # TODO: check the existence of extensions
        # Open fits files
        hdulist = fits.open(filename)
        # TODO: don't hardcode extension numbers and names here ... pass on from gp-cwt
        self.set_data(hdulist[0].data, hdulist['NormOffMap'].data)
        self.header = hdulist[0].header
        self.wcs = WCS(self.header)

    def do_transform(self):
        """Do the transform itself."""
        # TODO: after unit tests are added switch to astropy fftconvolve here.
        from scipy.signal import fftconvolve
        total_background = self.model + self.background + self.approx
        excess = self.image - total_background
        for key, kern in self.kernbase.items():
            self.transform[key] = fftconvolve(excess, kern, mode='same')
            self.error[key] = np.sqrt(fftconvolve(total_background, kern ** 2, mode='same'))

        self.approx = fftconvolve(self.image - self.model - self.bkg,
                                  self.kern_approx, mode='same')
        self.approx_bkg = fftconvolve(self.bkg, self.kern_approx, mode='same')

    def compute_support_peak(self, nsigma=2.0, nsigmap=4.0, remove_isolated=True):
        """Compute the multiresolution support with hard sigma clipping.

        Imposing a minimum significance on a connex region of significant pixels
        (i.e. source detection)
        """ 
        from scipy.ndimage import label
        # TODO: check that transform has been performed
        sig = self.transform / self.error
        for key in self.scales.keys():
            tmp = sig[key] > nsigma
            # produce a list of connex structures in the support
            l, n = label(tmp)
            for id in range(1, n):
                index = np.where(l == id)
                if remove_isolated:
                    if index[0].size == 1:
                        tmp[index] *= 0.0  # Remove isolated pixels from support
                signif = sig[key][index]
                if signif.max() < nsigmap:  # Remove significant pixels island from support 
                    tmp[index] *= 0.0  # if max does not reach maximal significance

            self.support[key] += tmp
            self.support[key] = self.support[key] > 0.

    def inverse_transform(self):
        """Do the inverse transform (reconstruct the image)."""
        res = np.sum(self.support * self.transform, 0)
        self.filter += res * (res > 0)
        self.model = self.filter
        return res

    def iterative_filter_peak(self, nsigma=3.0, nsigmap=4.0, niter=2, convergence=1e-5):
        """Run iterative filter peak algorithm."""
        var_ratio = 0.0
        for iiter in range(niter):
            self.do_transform()
            self.compute_support_peak(nsigma, nsigmap)
            res = self.inverse_transform()
            residual = self.image - (self.model + self.approx)
            tmp_var = residual.var()
            if iiter > 0:
                var_ratio = abs((self.residual_var - tmp_var) / self.residual_var)
                if var_ratio < convergence:
                    logging.info("Convergence reached at iteration {0}".format(iiter + 1))
                    return res
            self.residual_var = tmp_var
        logging.info("Convergence not formally reached at iteration {0}".format(iiter + 1))
        logging.info("Final convergence parameter {0}. Objective was {1}."
                     "".format(convergence, var_ratio)) 
        return res

    def max_scale_image(self):
        """Compute the maximum scale image."""
        maximum = np.argmax(self.transform, 0)
        return self.scale_array[maximum] * (self.support.sum(0) > 0)

    def save_filter(self, filename, clobber=False):
        """Save filter to file."""
        hdu = fits.PrimaryHDU(self.filter, self.header)
        hdu.writeto(filename, clobber=clobber)
        fits.append(filename, self.approx, self.header)
        fits.append(filename, self.filter + self.approx, self.header)
        fits.append(filename, self.max_scale_image(), self.header)

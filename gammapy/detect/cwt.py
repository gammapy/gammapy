# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.io import fits

__all__ = [
    'CWT',
]

log = logging.getLogger(__name__)


def gauss_kernel(radius, n_sigmas=8):
    """Normalized 2D gauss kernel array.

    TODO: replace by http://astropy.readthedocs.io/en/latest/api/astropy.convolution.Gaussian2DKernel.html
    once there are tests in place that establish the algorithm
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

    TODO: replace by http://astropy.readthedocs.io/en/latest/api/astropy.convolution.MexicanHat2DKernel.html
    once there are tests in place that establish the algorithm
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

    TODO: instead of storing all the arrays as data members,
    we could have a ``.data`` member which is a dict of images?

    TODO: give references

    Initialization of wavelet family.

    Data members used by this algorithm:

    - counts : 2D counts image (input, fixed)
    - background : 2D background image (input, fixed)

    - scales : dict
        - Keys are scale index integers
        - Values are scale values (pixel size)
    - kernbase : dict
        - Keys are scale index integers
        - Values are 2D kernel arrays (mexican hat)
    - kern_approx : 2D kernel array
        - Gaussian kernel from maximum scale

    The ``do_transform`` step computes the following:

    - transform : 3D cube, init to 0
      - Convolution of ``excess`` with kernel for each scale
    - error : 3D cube, init to 0
      - Convolution of ``total_background`` with kernel^2 for each scale
    - approx : 2D image, init to 0
      - Convolution of ``counts - model - background`` with ``kern_approx``
    - approx_bkg : 2D image, filled by do_transform
      - Convolution of ``background`` with ``kern_approx``

    - ``total_background = self.model + self.background + self.approx``

    The ``compute_support_peak`` step does the following:

    - computes significance ``sig = transform / error``
    - support : 3D cube exclusion mask
      - filled as ``sig > nsigma``

    The ``inverse_transform`` step does the following:

    - model : 2D image, init to 0
        - Fill ``res = np.sum(self.support * self.transform, axis=0)``
        - Fill ``self.model += res * (res > 0)``
        - What is this??


    - max_scale_image : 2D image, value = index of scale where emission is most significant

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
        for idx_scale in np.arange(0, nscales):
            scale = min_scale * scale_step ** idx_scale
            self.scales[idx_scale] = scale
            self.kernbase[idx_scale] = difference_of_gauss_kernel(scale, scale_step)

        max_scale = min_scale * scale_step ** nscales
        self.kern_approx = gauss_kernel(max_scale)

    def set_data(self, counts, background):
        """Set input images."""
        self.counts = np.array(counts, dtype=float)
        self.background = np.array(background, dtype=float)

        shape_2d = self.counts.shape
        self.model = np.zeros(shape_2d)
        self.approx = np.zeros(shape_2d)

        shape_3d = self.nscale, shape_2d[0], shape_2d[1]
        self.transform = np.zeros(shape_3d)
        self.error = np.zeros(shape_3d)
        self.support = np.zeros(shape_3d)

    def do_transform(self):
        """Do the transform itself.

        TODO: document. rename?
        """
        from scipy.signal import fftconvolve
        total_background = self.model + self.background + self.approx
        excess = self.counts - total_background

        for idx_scale, kern in self.kernbase.items():
            log.info('Computing transform and error')
            self.transform[idx_scale] = fftconvolve(excess, kern, mode='same')
            self.error[idx_scale] = np.sqrt(fftconvolve(total_background, kern ** 2, mode='same'))

        self.approx = fftconvolve(self.counts - self.model - self.background,
                                  self.kern_approx, mode='same')
        self.approx_bkg = fftconvolve(self.background, self.kern_approx, mode='same')

    def compute_support(self, nsigma=2.0, nsigmap=4.0, remove_isolated=True):
        """Compute the multiresolution support with hard sigma clipping.

        Imposing a minimum significance on a connected region of significant pixels
        (i.e. source detection)

        Parameters
        ----------
        TODO: document

        Returns
        -------
        TODO: for now it's stored as `self.support`. Maybe return
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

        return self.support

    def inverse_transform(self):
        """Do the inverse transform (reconstruct the image).

        TODO: describe better what this does.
        """
        res = np.sum(self.support * self.transform, axis=0)
        self.model += res * (res > 0)
        return res

    def run_one_iteration(self, nsigma, nsigmap):
        """TODO: document what this does.

        Parameters
        ----------
        TODO
        """
        self.do_transform()
        self.compute_support(nsigma, nsigmap)
        res = self.inverse_transform()
        return res

    def run_iteratively(self, nsigma=3.0, nsigmap=4.0, niter=2, convergence=1e-5):
        """Run iterative filter peak algorithm.

        Parameters
        ----------
        TODO
        """
        var_ratio = 0.0
        for iiter in range(niter):
            res = self.run_one_iteration(nsigma, nsigmap)

            # This is a check whether the iteration has converged.
            # TODO: document metric used, but not super important.
            # TODO: refactor check into extra method to make it easier to understand.
            residual = self.counts - (self.model + self.approx)
            tmp_var = residual.var()
            if iiter > 0:
                var_ratio = abs((self.residual_var - tmp_var) / self.residual_var)
                if var_ratio < convergence:
                    log.info("Convergence reached at iteration {0}".format(iiter + 1))
                    return res
            self.residual_var = tmp_var

        log.info("Convergence not formally reached at iteration {0}".format(iiter + 1))
        log.info("Final convergence parameter {0}. Objective was {1}."
                 "".format(convergence, var_ratio))
        return res

    @property
    def max_scale_image(self):
        """Compute the maximum scale image.

        TODO: document
        """
        scale_array = np.array(self.scales.values())
        maximum = np.argmax(self.transform, axis=0)
        return scale_array[maximum] * (self.support.sum(0) > 0)

    @property
    def model_plus_approx(self):
        """TODO: document what this is."""
        return self.model + self.approx

    def save_results(self, filename, header=None, overwrite=False):
        """Save results to file."""
        hdu_list = fits.HDUList()
        hdu_list.append(fits.PrimaryHDU())
        hdu_list.append(fits.ImageHDU(data=self.counts, header=header, name='counts'))
        hdu_list.append(fits.ImageHDU(data=self.background, header=header, name='background'))
        hdu_list.append(fits.ImageHDU(data=self.model, header=header, name='model'))
        hdu_list.append(fits.ImageHDU(data=self.approx, header=header, name='approx'))
        hdu_list.append(fits.ImageHDU(data=self.model_plus_approx, header=header, name='model_plus_approx'))

        # TODO: this isn't working yet. Why?
        # hdu_list.append(fits.ImageHDU(data=self.max_scale, header=self.header, name='max_scale'))

        hdu_list.writeto(filename, clobber=overwrite)

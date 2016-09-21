# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Ring background estimation.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from itertools import product
import numpy as np
from astropy.convolution import Ring2DKernel
from ..image import SkyImageList, SkyImage
from ..image.utils import scale_cube_fft


__all__ = [
    'AdaptiveRingBackgroundEstimator',
    'RingBackgroundEstimator',
    'ring_r_out',
    'ring_area_factor',
    'ring_alpha',
]


class AdaptiveRingBackgroundEstimator(object):
    """
    Adaptive ring background algorithm.

    This alogrithm extends the standard `RingBackground` method by adpating the
    size of the ring to achieve a minimum on / off exposure ratio (alpha) in regions
    where the area to estimate the background from is limited.

    Here is an illustration of the method:

    Parameters
    ----------
    r_in : float
        Inner radius of the ring.
    r_out_max : float
        Maximal outer radius of the ring.
    width : float
        Width of the ring.
    stepsize : float
        Stepsize used for increasing the radius.
    threshold : float
        Threshold above which the adaptive ring takes action.
    method : {'const. width', 'const. r_in'}
        Adaptive ring method.
    theta : float
        Integration radius used for alpha computation.
    """
    def __init__(self, r_in=None, r_out_max=None, width=None, stepsize=None,
                 threshold=None, method='const. width', theta=0.22):

        # default values
        stepsize = stepsize if stepsize else 0.02

        # input validation
        if method not in ['const. width', 'const. r_in']:
            raise ValueError("Not a valid adaptive ring method.")

        self.parameters = dict(r_in=r_in, r_out_max=r_out_max, width=width,
                               stepsize=stepsize, threshold=threshold,
                               method=method, theta=theta)


    def ring_kernels(self, image):
        """
        Ring kernel according to the specified method.

        Parameters
        ----------
        image : `~gammapy.image.SkyImage`
            Sky image specifying the WCS information.

        Returns
        -------
        kernels : list
            List of `~astropy.convolution.Kernel`
        """
        p = self.parameters
        r_ins, r_out_max = p['r_in'], p['r_out_max']
        widths, stepsize = p['width'], p['stepsize']

        kernels = []

        scale = image.wcs_pixel_scale()[0]

        if p['method'] == 'const. width':
            r_ins = np.arange(r_in, r_out_max - width, stepsize) / scale

        elif p['method'] == 'const. r_in':
            width = np.arange(width, r_out_max - r_in, stepsize) / scale

        for r_in, width in product(r_ins, widths):
            kernel = Ring2DKernel(r_in, width)
            kernel.normalize('peak')
            kernels.append(kernel)
        return kernels

    def _compute_alpha_approx_cube(self):
        """
        Compute alpha as on_exposure / off_exposure. Where off_exposure < 0,
        alpha is set to infinity.
        """
        exposure_on = self._cubes['exposure_on']
        exposure_off = self._cubes['exposure_off']

        alpha_approx = np.where(exposure_off > 0, exposure_on / exposure_off, np.inf)
        self._cubes['alpha_approx'] = alpha_approx

    def _compute_exposure_off_cube(self, kernels):
        """
        Compute off exposure cube by convolving the on exposure with different
        ring kernels and stacking the data along the third dimension.
        """
        exposure  = self._images['onexposure'].data
        exclusion = self._images['exclusion'].data
        exposure_off_cube = scale_cube_fft(exposure * exclusion, kernels)
        self._cubes['exposure_off'] = exposure_off_cube

    def _compute_exposure_on_cube(self, kernels, theta):
        """
        Compute on exposure cube, by convolving the on exposure with a tophat
        of radius theta, and stacking all images along the third dimension.
        """
        from scipy.ndimage import convolve
        exposure  = self._images['onexposure'].data

        tophat = Tophat2DKernel(theta)
        tophat.normalize('peak')
        exposure_on = convolve(exposure, tophat.array)
        exposure_on_cube = np.repeat(exposure_on[:, :, np.newaxis], len(kernels), axis=2)
        self._cubes['exposure_on'] = exposure_on_cube

    def _compute_off_cube(self, kernels):
        """
        Compute off cube by convolving the raw counts with different ring kernels
        and stacking the data along the third dimension.
        """
        counts = self._images['counts'].data
        exclusion = self._images['exclusion'].data
        off = scale_cube_fft(counts * exclusion, kernels)
        self._cubes['off'] = off

    def _reduce_cubes(self, threshold):
        """
        Compute off and off exposure map, by reducing the cubes. The data is
        iterated along the third axis (i.e. increasing ring sizes), the value
        with the first approximate alpha < threshold is taken.
        """
        alpha_approx_cube = self._cubes['alpha_approx']
        off_cube = self._cubes['off']
        exposure_off_cube = self._cubes['exposure_off']

        shape = alpha_approx_cube.shape[:2]
        off = np.tile(np.nan, shape)
        exposure_off = np.tile(np.nan, shape)

        for idx in np.arange(alpha_approx_cube.shape[-1]):
            mask = (alpha_approx_cube[:, :, idx] <= threshold) & np.isnan(off)
            off[mask] = off_cube[:, :, idx][mask]
            exposure_off[mask] = exposure_off_cube[:, :, idx][mask]
        self._images['off'] = off
        self._images['off_exposure'] = exposure_off

    def run(self, images):
        """
        Run adaptive ring background algorithm.

        Required sky images: {required}

        Parameters
        ----------
        images : `SkyImageList`
            Input sky images.

        Returns
        -------
        result : `SkyImageList`
            Result sky images.
        """
        images.check_required(['counts', 'exposure_on', 'exclusion'])

        p = self.parameters
        self._images = images
        self._cubes = {}
        wcs = images['counts'].wcs.copy()

        result = SkyImageList()

        kernels = self.ring_kernels(wcs)

        self._compute_exposure_on_cube(kernels, p['theta'])
        self._compute_exposure_off_cube(kernels)
        self._compute_off_cube(kernels)
        self._compute_alpha_approx_cube()
        self._reduce_cubes(p['threshold'])

        exposure_on = images['onexposure'].data
        exposure_off = self._images['off_exposure'].data
        off = self._images['off'].data
        alpha = exposure_on / exposure_off
        background =  alpha * off
        return SkyImageList(off=off, alpha=alpha, background=background,
                                  exposure_off=exposure_off, exposure_on=exposure_on,
                                  wcs=wcs)

class RingBackgroundEstimator(object):
    """Ring background method for cartesian coordinates.

    Step 1: apply exclusion mask
    Step 2: ring-correlate
    Step 3: apply psi cut

    TODO: add method to apply the psi cut

    Parameters
    ----------
    r_in : `~astropy.units.Quantity`
        Inner ring radius
    width : `~astropy.units.Quantity`
        Ring width.


    Examples
    --------
    Here's an example how to use the `RingBackgroundEstimator`:

        from astropy import units as u
        from gammapy.background import RingBackgroundEstimator
        from gammapy.image import SkyImageList

        filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/poisson_stats_image/input_all.fits.gz'
        images = SkyImageList.read(filename)
        images['exposure'].name = 'exposure_on'
        ring_bkg = RingBackgroundEstimator(r_in=0.35 * u.deg, width=0.3 * u.deg)
        results = ring_bkg.run(images)
        results['background'].show()


    See Also
    --------
    KernelBackgroundEstimator
    """
    def __init__(self, r_in, width):
        self.parameters = dict(r_in=r_in, width=width)

    def ring_convolve(self, image, **kwargs):
        """
        Convolve sky image with ring kernel.

        Parameters
        ----------
        image : `gammapy.image.SkyImage`
            Image
        **kwargs : dict
            Keyword arguments passed to `gammapy.image.SkyImage.convolve`
        """
        p = self.parameters

        scale = image.wcs_pixel_scale()[0]
        r_in = p['r_in'].to('deg') / scale
        width = p['width'].to('deg') / scale

        ring = Ring2DKernel(r_in.value, width.value)
        ring.normalize('peak')
        return image.convolve(ring.array, **kwargs)

    def run(self, images):
        """
        Run ring background algorithm.

        Required sky images: {required}

        Parameters
        ----------
        images : `SkyImageList`
            Input sky images.

        Returns
        -------
        result : `SkyImageList`
            Result sky images
        """
        images.check_required(['counts', 'exposure_on', 'exclusion'])
        p = self.parameters

        counts = images['counts']
        exclusion = images['exclusion']
        exposure_on = images['exposure_on']
        wcs = counts.wcs.copy()

        result = SkyImageList()

        counts_excluded = SkyImage(data=counts.data * exclusion.data, wcs=wcs)
        result['off'] = self.ring_convolve(counts_excluded)

        exposure_on_excluded = SkyImage(data=exposure_on.data * exclusion.data, wcs=wcs)
        result['exposure_off'] = self.ring_convolve(exposure_on_excluded)

        result['alpha'] = SkyImage(data=exposure_on.data / result['exposure_off'].data, wcs=wcs)
        result['background'] = SkyImage(data=result['alpha'].data * result['off'].data, wcs=wcs)
        return result

    def info(self):
        """
        Print summary info about the parameters.
        """
        print(str(self))

    def __str__(self):
        """
        String representation of the class.
        """
        info = "RingBackground parameters: \n"
        info += 'r_in : {}\n'.format(self.parameters['r_in'])
        info += 'width: {}\n'.format(self.parameters['width'])
        return info


def ring_r_out(theta, r_in, area_factor):
    """Compute ring outer radius.

    The determining equation is:
        area_factor =
        off_area / on_area =
        (pi (r_out**2 - r_in**2)) / (pi * theta**2 )

    Parameters
    ----------
    theta : float
        On region radius
    r_in : float
        Inner ring radius
    area_factor : float
        Desired off / on area ratio

    Returns
    -------
    r_out : float
        Outer ring radius
    """
    return np.sqrt(area_factor * theta ** 2 + r_in ** 2)


def ring_area_factor(theta, r_in, r_out):
    """Compute ring area factor.

    Parameters
    ----------
    theta : float
        On region radius
    r_in : float
        Inner ring radius
    r_out : float
        Outer ring radius
    """
    return (r_out ** 2 - r_in ** 2) / theta ** 2


def ring_alpha(theta, r_in, r_out):
    """Compute ring alpha, the inverse area factor.

    Parameters
    ----------
    theta : float
        On region radius
    r_in : float
        Inner ring radius
    r_out : float
        Outer ring radius
    """
    return 1. / ring_area_factor(theta, r_in, r_out)

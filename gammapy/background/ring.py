# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Ring background estimation."""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
from itertools import product
import numpy as np
from astropy.convolution import Ring2DKernel, Tophat2DKernel
import astropy.units as u
from ..image import SkyImageList, SkyImage
from ..image.utils import scale_cube

__all__ = [
    'AdaptiveRingBackgroundEstimator',
    'RingBackgroundEstimator',
    'ring_r_out',
    'ring_area_factor',
    'ring_alpha',
]


class AdaptiveRingBackgroundEstimator(object):
    """Adaptive ring background algorithm.

    This algorithm extends the standard `RingBackground` method by adapting the
    size of the ring to achieve a minimum on / off exposure ratio (alpha) in regions
    where the area to estimate the background from is limited.

    Parameters
    ----------
    r_in : `~astropy.units.Quantity`
        Inner radius of the ring.
    r_out_max : `~astropy.units.Quantity`
        Maximal outer radius of the ring.
    width : `~astropy.units.Quantity`
        Width of the ring.
    stepsize : `~astropy.units.Quantity`
        Stepsize used for increasing the radius.
    threshold_alpha : float
        Threshold on alpha above which the adaptive ring takes action.
    theta : `~astropy.units.Quantity`
        Integration radius used for alpha computation.
    method : {'fixed_width', 'fixed_r_in'}
        Adaptive ring method.

    Examples
    --------
    Here's an example how to use the `AdaptiveRingBackgroundEstimator`:

    .. code:: python

        from astropy import units as u
        from gammapy.background import AdaptiveRingBackgroundEstimator
        from gammapy.image import SkyImageList

        filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/poisson_stats_image/input_all.fits.gz'
        images = SkyImageList.read(filename)
        images['exposure'].name = 'exposure_on'
        adaptive_ring_bkg = RingBackgroundEstimator(r_in=0.22 * u.deg,
                                                    r_out_max=0.8 * u.deg,
                                                    width=0.1 * u.deg)
        results = adaptive_ring_bkg.run(images)
        results['background'].show()

    See Also
    --------
    RingBackgroundEstimator, gammapy.detect.KernelBackgroundEstimator
    """

    def __init__(self, r_in, r_out_max, width, stepsize=0.02 * u.deg,
                 threshold_alpha=0.1, theta=0.22 * u.deg, method='fixed_width'):
        # input validation
        if method not in ['fixed_width', 'fixed_r_in']:
            raise ValueError("Not a valid adaptive ring method.")

        self.parameters = OrderedDict(r_in=r_in, r_out_max=r_out_max, width=width,
                                      stepsize=stepsize, threshold_alpha=threshold_alpha,
                                      method=method, theta=theta)

    def kernels(self, image):
        """Ring kernels according to the specified method.

        Parameters
        ----------
        image : `~gammapy.image.SkyImage`
            Sky image specifying the WCS information.

        Returns
        -------
        kernels : list
            List of `~astropy.convolution.Ring2DKernel`
        """
        p = self.parameters

        scale = image.wcs_pixel_scale()[0]
        r_in = p['r_in'].to('deg') / scale
        r_out_max = p['r_out_max'].to('deg') / scale
        width = p['width'].to('deg') / scale
        stepsize = p['stepsize'].to('deg') / scale

        kernels = []

        if p['method'] == 'fixed_width':
            r_ins = np.arange(r_in.value, (r_out_max - width).value, stepsize.value)
            widths = [width.value]
        elif p['method'] == 'fixed_r_in':
            widths = np.arange(width.value, (r_out_max - r_in).value, stepsize.value)
            r_ins = [r_in.value]

        for r_in, width in product(r_ins, widths):
            kernel = Ring2DKernel(r_in, width)
            kernel.normalize('peak')
            kernels.append(kernel)
        return kernels

    def _alpha_approx_cube(self, cubes):
        """Compute alpha as on_exposure / off_exposure.

        Where off_exposure < 0, alpha is set to infinity.
        """
        exposure_on = cubes['exposure_on']
        exposure_off = cubes['exposure_off']

        alpha_approx = np.where(exposure_off > 0, exposure_on / exposure_off, np.inf)
        return alpha_approx

    def _exposure_off_cube(self, images, kernels):
        """Compute off exposure cube.

        The on exposure is convolved with different
        ring kernels and stacking the data along the third dimension.
        """
        exposure = images['exposure_on'].data
        exclusion = images['exclusion'].data
        return scale_cube(exposure * exclusion, kernels)

    def _exposure_on_cube(self, images, kernels):
        """Compute on exposure cube.

        Calculated by convolving the on exposure with a tophat
        of radius theta, and stacking all images along the third dimension.
        """
        from scipy.ndimage import convolve

        exposure_on = images['exposure_on']
        scale = exposure_on.wcs_pixel_scale()[0]
        theta = self.parameters['theta'] * scale

        tophat = Tophat2DKernel(theta.value)
        tophat.normalize('peak')
        exposure_on = convolve(exposure_on, tophat.array)
        exposure_on_cube = np.repeat(exposure_on[:, :, np.newaxis], len(kernels), axis=2)
        return exposure_on_cube

    def _off_cube(self, images, kernels):
        """Compute off cube.

        Calculated by convolving the raw counts with different ring kernels
        and stacking the data along the third dimension.
        """
        counts = images['counts'].data
        exclusion = images['exclusion'].data
        return scale_cube(counts * exclusion, kernels)

    def _reduce_cubes(self, cubes):
        """Compute off and off exposure map.

        Calulated by reducing the cubes. The data is
        iterated along the third axis (i.e. increasing ring sizes), the value
        with the first approximate alpha < threshold is taken.
        """
        threshold = self.parameters['threshold_alpha']

        alpha_approx_cube = cubes['alpha_approx']
        off_cube = cubes['off']
        exposure_off_cube = cubes['exposure_off']

        shape = alpha_approx_cube.shape[:2]
        off = np.tile(np.nan, shape)
        exposure_off = np.tile(np.nan, shape)

        for idx in np.arange(alpha_approx_cube.shape[-1]):
            mask = (alpha_approx_cube[:, :, idx] <= threshold) & np.isnan(off)
            off[mask] = off_cube[:, :, idx][mask]
            exposure_off[mask] = exposure_off_cube[:, :, idx][mask]

        return exposure_off, off

    def run(self, images):
        """Run adaptive ring background algorithm.

        Parameters
        ----------
        images : `SkyImageList`
            Input sky images.

        Returns
        -------
        result : `SkyImageList`
            Result sky images.
        """
        required = ['counts', 'exposure_on', 'exclusion']
        images.check_required(required)
        wcs = images['counts'].wcs.copy()

        cubes = OrderedDict()

        kernels = self.kernels(images['counts'])

        cubes['exposure_on'] = self._exposure_on_cube(images, kernels)
        cubes['exposure_off'] = self._exposure_off_cube(images, kernels)
        cubes['off'] = self._off_cube(images, kernels)
        cubes['alpha_approx'] = self._alpha_approx_cube(cubes)

        exposure_off, off = self._reduce_cubes(cubes)
        alpha = images['exposure_on'].data / exposure_off
        not_has_exposure = ~(images['exposure_on'].data > 0)

        # set data outside fov to zero
        for data in [alpha, off, exposure_off]:
            data[not_has_exposure] = 0

        background = alpha * off

        result = SkyImageList()
        result['exposure_off'] = SkyImage(data=exposure_off, wcs=wcs)
        result['off'] = SkyImage(data=off, wcs=wcs)
        result['alpha'] = SkyImage(data=alpha, wcs=wcs)
        result['background'] = SkyImage(data=background, wcs=wcs)
        return result


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
    use_fft_convolution : bool
        Use fft convolution.


    Examples
    --------
    Here's an example how to use the `RingBackgroundEstimator`:

    .. code:: python

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
    gammapy.detect.KernelBackgroundEstimator, AdaptiveRingBackgroundEstimator
    """

    def __init__(self, r_in, width, use_fft_convolution=False):
        self.parameters = dict(r_in=r_in, width=width, use_fft_convolution=use_fft_convolution)

    def kernel(self, image):
        """Ring kernel.

        Parameters
        ----------
        image : `gammapy.image.SkyImage`
            Image

        Returns
        -------
        ring : `~astropy.convolution.Ring2DKernel`
            Ring kernel.
        """
        p = self.parameters

        scale = image.wcs_pixel_scale()[0]
        r_in = p['r_in'].to('deg') / scale
        width = p['width'].to('deg') / scale

        ring = Ring2DKernel(r_in.value, width.value)
        ring.normalize('peak')
        return ring

    def run(self, images):
        """Run ring background algorithm.

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
        p = self.parameters
        required = ['counts', 'exposure_on', 'exclusion']
        images.check_required(required)
        counts, exposure_on, exclusion = [images[_] for _ in required]
        wcs = counts.wcs.copy()

        result = SkyImageList()
        ring = self.kernel(counts)

        counts_excluded = SkyImage(data=counts.data * exclusion.data, wcs=wcs)
        result['off'] = counts_excluded.convolve(ring.array, mode='constant',
                                                 use_fft=p['use_fft_convolution'])
        result['off'].data = result['off'].data.astype(int)

        exposure_on_excluded = SkyImage(data=exposure_on.data * exclusion.data, wcs=wcs)

        result['exposure_off'] = exposure_on_excluded.convolve(ring.array, mode='constant',
                                                               use_fft=p['use_fft_convolution'])

        with np.errstate(divide='ignore'):
            # set pixels, where ring is too small to NaN
            not_has_off_exposure = ~(result['exposure_off'].data > 0)
            result['exposure_off'].data[not_has_off_exposure] = np.nan

            result['alpha'] = SkyImage(data=exposure_on.data / result['exposure_off'].data, wcs=wcs)

            not_has_exposure = ~(exposure_on.data > 0)
            result['alpha'].data[not_has_exposure] = 0

        result['background'] = SkyImage(data=result['alpha'].data * result['off'].data, wcs=wcs)
        return result

    def info(self):
        """Print summary info about the parameters."""
        print(str(self))

    def __str__(self):
        """String representation of the class."""
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

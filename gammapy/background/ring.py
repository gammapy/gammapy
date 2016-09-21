# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Ring background estimation.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.convolution import Ring2DKernel
from ..image import SkyImageList, SkyImage


__all__ = [
    'RingBackgroundEstimator',
    'ring_r_out',
    'ring_area_factor',
    'ring_alpha',
]


class RingBackgroundEstimator(object):
    """
    Ring background method for cartesian coordinates.

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

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Ring background estimation.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.convolution import Ring2DKernel
from ..image import SkyImageList, required_skyimages


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
    """
    def __init__(self, r_in, width):
        self.parameters = dict(r_in=r_in, width=width)

    def ring_convolve(self, image, **kwargs):
        """
        Convolve sky image with ring kernel.

        Parameters
        ----------
        image : `SkyImage`
            Image
        """
        p = self.parameters

        scale = image.wcs_pixel_scale()[0]
        r_in = p['r_in'].to('deg') / scale
        width = p['width'].to('deg') / scale

        ring = Ring2DKernel(r_in.value, width.value)
        ring.normalize('peak')
        return image.convolve(ring, fft=True)

    @required_skyimages('counts', 'exposure_on', 'exclusion')
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
        p = self.parameters

        counts = images['counts']
        exclusion = images['exclusion']
        exposure_on = images['exposure_on']

        result = SkyImageList()

        result['off'] = self.ring_convolve(counts * exclusion)
        result['exposure_off'] = self.ring_convolve(exposure_on * exclusion)
        result['alpha'] = exposure_on / result['exposure_off']
        result['background'] = result['alpha'] * result['off']
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

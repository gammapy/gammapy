# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Ring background estimation.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from ..image import ring_correlate

__all__ = [
    'RingBackground',
    'ring_r_out',
    'ring_area_factor',
    'ring_alpha',
]


class RingBackground(object):
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
    r_out : `~astropy.units.Quantity`
        Outer ring radius
    """

    def __init__(self, r_in, r_out):
        if not r_in < r_out:
            raise ValueError('r_in must be smaller than r_out')
        self.parameters = dict(r_in=r_in, r_out=r_out)

    @required_skyimages('counts', 'onexposure', 'exclusion')
    def run(self, images):
        """
        Run ring background algorithm.

        Required sky images: {required}

        Parameters
        ----------
        images : `SkyImageCollection`
            Input sky images.

        Returns
        -------
        result : `SkyImageCollection`
            Result sky images
        """
        p = self.parameters
        self._images = images
        wcs = images.counts.wcs.copy()

        counts = images['counts'].data
        exposure_on = images['onexposure'].data
        exclusion = images['exclusion'].data

        off = ring_correlate(counts * exclusion, p.r_in, p.r_out)
        exposure_off = ring_correlate(exposure_on * exclusion, p.r_in, p.r_out)
        alpha = exposure_on / exposure_off
        background = alpha * off
        return SkyImageCollection(off=off, exposure_off=exposure_off,
                                  alpha=alpha, background=background, wcs=wcs)


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

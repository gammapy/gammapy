# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.wcs import WCS
from ..image import exclusion_distance

__all__ = [
    'ExclusionMask',
]

class ExclusionMask(object):
    """Exclusion mask

    Parameters
    ----------
    mask : `~numpy.ndarray`; dtype = int, bool
         Exclusion mask
    """
    def __init__(self, mask, wcs=None):
        self.mask = mask
        self.wcs = wcs
        self._distance_image = exclusion_distance(mask)

    @classmethod
    def from_hdu(cls, hdu):
        mask = np.array(hdu.data, dtype = int)
        wcs = WCS(hdu.header)
        return cls(mask, wcs)

    @property
    def distance_image(self):
        """Map containting the distance to the nearest exclusion region"""
        return self._distance_image

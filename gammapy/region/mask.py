# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.wcs import WCS

__all__ = [
    'ExclusionMask',
]


class ExclusionMask(object):

    def __init__(self, mask, wcs=None):
        self.mask = mask
        self.wcs = wcs

        self._distance_image = 42

    @classmethod
    def from_hdu(cls, hdu):
        mask = hdu.data
        wcs = WCS(hdu.header)
        return cls(mask, wcs)

    @property
    def distance_image(self):
        """..."""
        return self._distance_image

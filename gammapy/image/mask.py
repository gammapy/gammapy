# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.wcs import WCS
from ..image import exclusion_distance, lon_lat_circle_mask, coordinates

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
    def create_random(cls, hdu, n=4, max_rad=40):
        """Create random exclusion mask (n circles) on a  given image

        This is useful for testing

        Parameters
        ----------
        hdu : `~astropy.fits.ImageHDU`
            ImageHDU
        n : int
            Number of circles to place
        max_rad : int
            Maximum circle radius
        """
        
        wcs = WCS(hdu.header)
        ny,nx = hdu.data.shape
        mask = np.ones((nx,ny), dtype = int)
        xx = np.random.choice(np.arange(nx),n)
        yy = np.random.choice(np.arange(ny),n)
        rr = np.random.rand(n) * max_rad
        
        for x,y,r in zip(xx,yy,rr):
            xd, yd = np.ogrid[-x:nx-x, -y:ny-y] 
            val = xd * xd + yd * yd <= r * r
            mask[val] = 0

        return cls(mask, wcs)

    @classmethod
    def from_hdu(cls, hdu):
        """Read exclusion mask from ImageHDU

        Parameters
        ----------
        hdu : `~astropy.fits.ImageHDU`
            ImageHDU containing only an exlcusion mask (int, bool)
        """
        mask = np.array(hdu.data, dtype = int)
        wcs = WCS(hdu.header)
        return cls(mask, wcs)

    @property
    def distance_image(self):
        """Map containting the distance to the nearest exclusion region"""
        return self._distance_image


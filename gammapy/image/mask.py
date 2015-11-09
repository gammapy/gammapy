# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
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
        self._distance_image = None

    @classmethod
    def create_random(cls, hdu, n=4, min_rad=0, max_rad=40):
        """Create random exclusion mask (n circles) on a  given image

        This is useful for testing

        Parameters
        ----------
        hdu : `~astropy.fits.ImageHDU`
            ImageHDU
        n : int
            Number of circles to place
        min_rad : int
            Minimum circle radius in pixels
        max_rad : int
            Maximum circle radius in pixels
        """

        wcs = WCS(hdu.header)
        mask = np.ones(hdu.data.shape, dtype=int)
        nx, ny = mask.shape
        xx = np.random.choice(np.arange(nx), n)
        yy = np.random.choice(np.arange(ny), n)
        rr = min_rad + np.random.rand(n) * (max_rad - min_rad)

        for x, y, r in zip(xx, yy, rr):
            xd, yd = np.ogrid[-x:nx - x, -y:ny - y]
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
        mask = np.array(hdu.data, dtype=int)
        wcs = WCS(hdu.header)
        return cls(mask, wcs)

    @classmethod
    def from_fits(cls, excl_file):
        """Read exclusion mask fits file

        Parameters
        ----------
        excl_file : str
            fits file containing an ImageHDU
        """
        hdu = fits.open(excl_file)[0]
        return cls.from_hdu(hdu)

    @classmethod
    def from_ds9(cls, excl_file, hdu):
        """Create exclusion mask from ds9 regions file

        Uses the pyregion package
        (http://pyregion.readthedocs.org/en/latest/index.html)

        Parameters
        ----------
        excl_file : str
            ds9 region file
        hdu : `~astropy.fits.ImageHDU`
            Map to fill exclusion mask
        """
        import pyregion
        r = pyregion.open(excl_file)
        val = r.get_mask(hdu=hdu)
        mask = np.invert(val)
        wcs = WCS(hdu.header)
        return cls(mask, wcs)

    def to_hdu(self):
        """Create ImageHDU containting the exclusion mask
        """
        header = self.wcs.to_header()
        return fits.ImageHDU(self.mask, header)

    def plot(self, ax, **kwargs):
        """Plot

        Parameters
        ----------
        ax : `~astropy.wcsaxes.WCSAxes`
            WCS axis object
        """
        from matplotlib import colors
        import matplotlib.pyplot as plt
        if 'cmap' not in locals():
            cmap = colors.ListedColormap(['black', 'lightgrey'])
        ax.imshow(self.mask, cmap=cmap, origin='lower')

    @property
    def distance_image(self):
        """Map containting the distance to the nearest exclusion region"""

        if self._distance_image is None:
            self._distance_image = exclusion_distance(self.mask)

        return self._distance_image


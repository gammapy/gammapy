# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import Latitude, Longitude, Angle
from astropy.utils import lazyproperty

from ..image import (
    exclusion_distance,
    lon_lat_circle_mask,
)
from .maps import SkyMap


__all__ = [
    'ExclusionMask',
    'make_tevcat_exclusion_mask'
]


class ExclusionMask(SkyMap):
    """Exclusion mask

    """

    def fill_random_circles(self, n=4, min_rad=0, max_rad=40):
        """Create random exclusion mask (n circles) on a  given image

        This is useful for testing

        Parameters
        ----------
        n : int
            Number of circles to place
        min_rad : int
            Minimum circle radius in pixels
        max_rad : int
            Maximum circle radius in pixels
        """
        # TODO: is it worth to change this to take the radius in deg?
        mask = np.ones(self.data.shape, dtype=int)
        nx, ny = mask.shape
        xx = np.random.choice(np.arange(nx), n)
        yy = np.random.choice(np.arange(ny), n)
        rr = min_rad + np.random.rand(n) * (max_rad - min_rad)

        for x, y, r in zip(xx, yy, rr):
            xd, yd = np.ogrid[-x:nx - x, -y:ny - y]
            val = xd * xd + yd * yd <= r * r
            mask[val] = 0
        self.data = mask

    @classmethod
    def from_ds9(cls, excl_file, hdu):
        """Create exclusion mask from ds9 regions file

        Uses the pyregion package
        (http://pyregion.readthedocs.io/en/latest/index.html)

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
        return cls(data=mask, wcs=wcs)

    def plot(self, ax=None, fig=None, **kwargs):
        """Plot exclusion mask

        Parameters
        ----------
        ax : `~astropy.wcsaxes.WCSAxes`, optional
            WCS axis object

        Returns
        ----------
        ax : `~astropy.wcsaxes.WCSAxes`, optional
            WCS axis object
        """
        from matplotlib import colors

        kwargs.setdefault('cmap', colors.ListedColormap(['black', 'lightgrey']))
        kwargs.setdefault('origin', 'lower')

        super(ExclusionMask, self).plot(ax, fig, **kwargs)

    @lazyproperty
    def distance_image(self):
        """Map containting the distance to the nearest exclusion region."""
        return exclusion_distance(self.mask)

    # Set alias for mask
    # TODO: Add mask attribute to sky map class or rename self.mask to self.data
    @property
    def mask(self):
        return self.data

    # TODO: right now the extension name is hardcoded to 'exclusion', because
    # single image Fits file often contain a PrimaryHDU and an ImageHDU.
    # Is there a better / more flexible solution?
    @classmethod
    def read(cls, fobj, *args, **kwargs):
        # Check if extension name is given, else default to 'exclusion'
        kwargs['extname'] = kwargs.get('extname', 'exclusion')
        return super(ExclusionMask, cls).read(fobj, *args, **kwargs)


def make_tevcat_exclusion_mask():
    """Create an all-sky exclusion mask containing all TeVCat sources

    Returns
    -------
    mask : `~gammapy.image.ExclusionMask`
        Exclusion mask
    """

    # TODO: make this a method ExclusionMask.from_catalog()?
    from gammapy.catalog import load_catalog_tevcat

    tevcat = load_catalog_tevcat()
    all_sky_exclusion = ExclusionMask.empty(nxpix=3600, nypix=1800, binsz=0.1,
                                            fill=1, dtype='int')
    val_lon, val_lat = all_sky_exclusion.coordinates()
    lons = Longitude(val_lon, 'deg')
    lats = Latitude(val_lat, 'deg')

    for source in tevcat:
        lon = Longitude(source['coord_gal_lon'], 'deg')
        lat = Latitude(source['coord_gal_lat'], 'deg')
        x = Angle(source['size_x'], 'deg')
        y = Angle(source['size_y'], 'deg')
        if np.isnan(x) and np.isnan(y):
            rad = Angle('0.3 deg')
        else:
            rad = x if x > y else y

        mask = lon_lat_circle_mask(lons, lats, lon, lat, rad)
        all_sky_exclusion.data[mask] = 0

    return all_sky_exclusion

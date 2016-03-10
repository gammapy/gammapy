# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from ..utils.scripts import make_path
from astropy.coordinates import Latitude, Longitude, Angle
from astropy.utils import lazyproperty
from ..image import (
    exclusion_distance,
    lon_lat_circle_mask,
    coordinates,
    make_empty_image,
)
from .maps import SkyMap

__all__ = [
    'ExclusionMask',
    'make_tevcat_exclusion_mask'
]


class ExclusionMask(SkyMap):
    """Exclusion mask

    """

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

        return cls(data=mask, wcs=wcs)

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

        if 'cmap' not in locals():
            cmap = colors.ListedColormap(['black', 'lightgrey'])

        kwargs['cmap'] = cmap
        kwargs['origin'] = 'lower'
        super(ExclusionMask, self).plot(ax, fig, **kwargs)

    @lazyproperty
    def distance_image(self):
        """Map containting the distance to the nearest exclusion region"""
        return exclusion_distance(self.mask)

    # Set alias for mask
    # TODO: Add mask attribute to sky map class
    @property
    def mask(self):
        return self.data


def make_tevcat_exclusion_mask():
    """Create an all-sky exclusion mask containing all TeVCat sources
    
    Returns
    -------
    mask : `~gammapy.image.ExclusionMask`
        Exclusion mask
    """

    from gammapy.catalog import load_catalog_tevcat

    tevcat = load_catalog_tevcat()
    all_sky_exclusion = make_empty_image(nxpix=3600, nypix=1800, binsz=0.1)
    val = np.ones(shape=all_sky_exclusion.data.shape)
    all_sky_exclusion.data = val
    val_lon, val_lat = coordinates(all_sky_exclusion)
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

    return ExclusionMask.from_hdu(all_sky_exclusion)

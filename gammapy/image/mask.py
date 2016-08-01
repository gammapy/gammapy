# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import Latitude, Longitude, Angle
from astropy.utils import lazyproperty
from ..image import lon_lat_circle_mask
from .core import SkyImage

__all__ = [
    'SkyMask',
    'make_tevcat_exclusion_mask'
]


class SkyMask(SkyImage):
    """Sky image mask.

    `SkyMask` is a `~gammapy.image.SkyMap` sub-class, i.e. it inherits
    all of it's features. The distinction is that `SkyMask` is to
    represent boolean masks and has methods that only make sense for
    mask data. The data array can be integer or float, but if it is,
    it should only contain pixel values of 0 or 1.

    TODO: explain about semantics and give examples what 0 and 1 mean
    in different applications (or link to other docs).
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
        data = np.ones(self.data.shape, dtype=int)
        nx, ny = data.shape
        xx = np.random.choice(np.arange(nx), n)
        yy = np.random.choice(np.arange(ny), n)
        rr = min_rad + np.random.rand(n) * (max_rad - min_rad)

        for x, y, r in zip(xx, yy, rr):
            xd, yd = np.ogrid[-x:nx - x, -y:ny - y]
            val = xd * xd + yd * yd <= r * r
            data[val] = 0

        self.data = data

    def open(self, structure):
        """
        Binary opening with structuring element.

        Calls `scipy.ndimage.binary_opening`.

        Parameters
        ----------
        structure : array-like
            Structuring kernel. Must be boolean i.e. only contain 1 and 0 values.

        Returns
        -------
        skymask : `SkyMask`
            Opened sky mask.
        """
        from scipy.ndimage import binary_opening
        data = binary_opening(self.data, structure)
        return SkyMask(data=data, wcs=self.wcs)

    def dilate(self, structure):
        """
        Binary dilation with structuring element.

        Calls `scipy.ndimage.binary_dilation`.

        Parameters
        ----------
        structure : array-like
            Structuring kernel. Must be boolean i.e. only contain 1 and 0 values.

        Returns
        -------
        skymask : `SkyMask`
            Dilated sky mask.
        """
        from scipy.ndimage import binary_dilation
        data = binary_dilation(self.data, structure)
        return SkyMask(data=data, wcs=self.wcs)

    def close(self, structure):
        """
        Binary closing with structuring element.

        Calls `scipy.ndimage.binary_closing`.

        Parameters
        ----------
        structure : array-like
            Structuring kernel. Must be boolean i.e. only contain 1 and 0 values.

        Returns
        -------
        skymask : `SkyMask`
            Closed sky mask.
        """
        from scipy.ndimage import binary_closing
        data = binary_closing(self.data, structure)
        return SkyMask(data=data, wcs=self.wcs)

    def erode(self, structure):
        """
        Binary erosion with structuring element.

        Calls `scipy.ndimage.binary_erosion`.

        Parameters
        ----------
        structure : array-like
            Structuring kernel. Must be boolean i.e. only contain 1 and 0 values.

        Returns
        -------
        skymask : `SkyMask`
            Eroded sky mask.
        """
        from scipy.ndimage import binary_erosion
        data = binary_erosion(self.data, structure)
        return SkyMask(data=data, wcs=self.wcs)

    def plot(self, ax=None, fig=None, **kwargs):
        """Plot exclusion mask

        Parameters
        ----------
        ax : `~astropy.wcsaxes.WCSAxes`, optional
            WCS axis object

        Returns
        -------
        ax : `~astropy.wcsaxes.WCSAxes`, optional
            WCS axis object
        """
        from matplotlib import colors

        kwargs.setdefault('cmap', colors.ListedColormap(['black', 'lightgrey']))

        super(SkyMask, self).plot(ax, fig, **kwargs)

    @lazyproperty
    def distance_image(self):
        """Distance to nearest exclusion region.

        Compute distance map, i.e. the Euclidean (=Cartesian 2D)
        distance (in pixels) to the nearest exclusion region.

        We need to call distance_transform_edt twice because it only computes
        dist for pixels outside exclusion regions, so to get the
        distances for pixels inside we call it on the inverted mask
        and then combine both distance images into one, using negative
        distances (note the minus sign) for pixels inside exclusion regions.

        If data consist only of ones, it'll be supposed to be far away
        from zero pixels, so in capacity of answer it should be return
        the matrix with the shape as like as data but packed by constant
        value Max_Value (MAX_VALUE = 1e10).

        If data consist only of zeros, it'll be supposed to be deep inside
        an exclusion region, so in capacity of answer it should be return
        the matrix with the shape as like as data but packed by constant
        value -Max_Value (MAX_VALUE = 1e10).

        Returns
        -------
        distance : `~gammapy.image.SkyImage`
            Sky map of distance to nearest exclusion region.

        Examples
        --------
        >>> from gammapy.image import SkyMask
        >>> data = np.array([[0., 0., 1.], [1., 1., 1.]])
        >>> mask = SkyMask(data=data)
        >>> print(mask.distance_image.data)
        [[-1, -1, 1], [1, 1, 1.41421356]]
        """
        from scipy.ndimage import distance_transform_edt

        max_value = 1e10

        if np.all(self.data == 1):
            return SkyImage.empty_like(self, fill=max_value)

        if np.all(self.data == 0):
            return SkyImage.empty_like(self, fill=-max_value)

        distance_outside = distance_transform_edt(self.data)

        invert_mask = np.invert(np.array(self.data, dtype=np.bool))
        distance_inside = distance_transform_edt(invert_mask)

        distance = np.where(self.data, distance_outside, -distance_inside)

        return SkyImage(data=distance, wcs=self.wcs)

    # TODO: right now the extension name is hardcoded to 'exclusion', because
    # single image Fits file often contain a PrimaryHDU and an ImageHDU.
    # Is there a better / more flexible solution?
    @classmethod
    def read(cls, fobj, *args, **kwargs):
        # Check if extension name is given, else default to 'exclusion'
        kwargs['extname'] = kwargs.get('extname', 'exclusion')
        return super(SkyMask, cls).read(fobj, *args, **kwargs)


def make_tevcat_exclusion_mask():
    """Create an all-sky exclusion mask containing all TeVCat sources

    Returns
    -------
    mask : `~gammapy.image.SkyMask`
        Exclusion mask
    """

    # TODO: make this a method SkyMask.from_catalog()?
    from gammapy.catalog import load_catalog_tevcat

    tevcat = load_catalog_tevcat()
    all_sky_exclusion = SkyMask.empty(nxpix=3600, nypix=1800, binsz=0.1,
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

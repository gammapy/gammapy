# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
import numpy as np
from astropy.extern import six
from astropy.utils.misc import InheritDocstrings
from astropy.io import fits
from .geom import pix_tuple_to_idx

__all__ = [
    'MapBase',
]


class MapMeta(InheritDocstrings, abc.ABCMeta):
    pass


@six.add_metaclass(MapMeta)
class MapBase(object):
    """Abstract map class.

    This can represent WCS- or HEALPIX-based maps
    with 2 spatial dimensions and N non-spatial dimensions.

    Parameters
    ----------
    geom : `~gammapy.maps.geom.MapGeom`

    data : `~numpy.ndarray`
    """

    def __init__(self, geom, data):
        self._data = data
        self._geom = geom

    @property
    def data(self):
        """Array of data values."""
        return self._data

    @property
    def geom(self):
        return self._geom

    @data.setter
    def data(self, val):
        if val.shape != self.data.shape:
            raise Exception('Wrong shape.')
        self._data = val

    @classmethod
    def create(cls, **kwargs):
        """Create an empty map object.  This method accepts generic options
        listed below as well as options for `~HpxMap` and `~WcsMap`
        objects (see `~HpxMap.create` and `~WcsMap.create` for WCS-
        and HPX-specific options).

        Parameters
        ----------
        coordsys : str
            Coordinate system, either Galactic ('GAL') or Equatorial
            ('CEL').

        map_type : str
            Internal map representation.  Valid types are `wcs`,
            `wcs-sparse`,`hpx`, and `hpx-sparse`.

        binsz : float or `~numpy.ndarray`
            Pixel size in degrees.

        skydir : `~astropy.coordinates.SkyCoord`
            Coordinate of map center.

        axes : list
            List of `~MapAxis` objects for each non-spatial dimension.
            If None then the map will be a 2D image.

        dtype : str
            Data type, default is float32

        unit : str or `~astropy.units.Unit`
            Data unit.

        Returns
        -------
        map : `~MapBase`
            Empty map object.

        """

        from .hpxmap import HpxMap
        from .wcsmap import WcsMap

        map_type = kwargs.setdefault('map_type', 'wcs')

        if 'wcs' in map_type.lower():
            return WcsMap.create(**kwargs)
        elif 'hpx' in map_type.lower():
            return HpxMap.create(**kwargs)
        else:
            raise ValueError('Unrecognized map type: {}'.format(map_type))

    @classmethod
    def read(cls, filename, **kwargs):
        """Read a map from a FITS file.

        Parameters
        ----------
        filename : str
            Name of the FITS file.
        hdu : str
            Name or index of the HDU with the map data.
        hdu_bands : str
            Name or index of the HDU with the BANDS table.  If not
            defined this will be inferred from the FITS header of the
            map HDU.

        Returns
        -------
        map_out : `~MapBase`
            Map object

        """
        with fits.open(filename) as hdulist:
            map_out = cls.from_hdulist(hdulist, **kwargs)
        return map_out

    def write(self, filename, **kwargs):
        """Write to a FITS file.

        Parameters
        ----------
        filename : str
            Output file name.
        extname : str
            Set the name of the image extension.  By default this will
            be set to SKYMAP (for BINTABLE HDU) or PRIMARY (for IMAGE
            HDU).
        extname_bands : str
            Set the name of the bands table extension.  By default this will
            be set to BANDS.
        hpxconv : str
            Format convention for HEALPix maps.  This option can be used to
            write files that are compliant with non-standard HEALPix
            conventions.
        sparse : bool
            Sparsify the map by dropping pixels with zero amplitude.

        """
        hdulist = self.to_hdulist(**kwargs)
        overwrite = kwargs.get('overwrite', True)
        hdulist.writeto(filename, overwrite=overwrite)

    @abc.abstractmethod
    def iter_by_image(self):
        """Iterate over image planes of the map returning a tuple with the image
        array and image plane index.

        Returns
        -------
        val : `~np.ndarray`
            Array of image plane values.
        idx : tuple
            Index of image plane.

        """
        pass

    @abc.abstractmethod
    def iter_by_pix(self, buffersize=1):
        """Iterate over elements of the map returning a tuple with values and
        pixel coordinates.

        Parameters
        ----------
        buffersize : int
            Set the size of the buffer.  The map will be returned in
            chunks of the given size.

        Returns
        -------
        val : `~np.ndarray`
            Map values.
        pix : tuple
            Tuple of pixel coordinates.
        """
        pass

    @abc.abstractmethod
    def iter_by_coords(self, buffersize=1):
        """Iterate over elements of the map returning a tuple with values and
        map coordinates.

        Parameters
        ----------
        buffersize : int
            Set the size of the buffer.  The map will be returned in
            chunks of the given size.

        Returns
        -------
        val : `~np.ndarray`
            Map values.
        coords : tuple
            Tuple of map coordinates.
        """
        pass

    @abc.abstractmethod
    def sum_over_axes(self):
        """Reduce to a 2D image by summing over non-spatial dimensions."""
        pass

    def reproject(self, geom, order=1, mode='interp'):
        """Reproject this map to a different geometry.

        Parameters
        ----------
        geom : `~MapGeom`
            Geometry of projection.

        mode : str
            Method for reprojection.  'interp' method interpolates at pixel
            centers.  'exact' method integrates over intersection of pixels.

        order : int or str
            Order of interpolating polynomial (0 = nearest-neighbor, 1 =
            linear, 2 = quadratic, 3 = cubic).

        Returns
        -------
        map : `~MapBase`
            Reprojected map.

        """
        if geom.ndim == 2 and self.geom.ndim > 2:
            geom = geom.to_cube(self.geom.axes)
        elif geom.ndim != self.geom.ndim:
            raise ValueError('Projection geometry must be 2D or have the '
                             'same number of dimensions as the map.')

        if geom.projection == 'HPX':
            return self._reproject_hpx(geom, mode=mode, order=order)
        else:
            return self._reproject_wcs(geom, mode=mode, order=order)

    @abc.abstractmethod
    def pad(self, pad_width):
        """Pad the spatial dimension of the map by extending the edge of the
        map by the given number of pixels.

        Parameters
        ----------
        pad_width : {sequence, array_like, int}
            Number of values padded to the edges of each axis, passed to `numpy.pad`

        Returns
        -------
        map : `~MapBase`
            Padded map.
        """
        pass

    @abc.abstractmethod
    def crop(self, crop_width):
        """Crop the spatial dimension of the map.

        Parameters
        ----------
        crop_width : {sequence, array_like, int}
            Number of values cropped from the edges of each axis.
            Defined analogously to `pad_with` from `~numpy.pad`.

        Returns
        -------
        map : `~MapBase`
            Cropped map.
        """
        pass

    @abc.abstractmethod
    def downsample(self, factor):
        """Downsample the spatial dimension of the map by a given factor. 

        Parameters
        ----------
        factor : int
            Downsampling factor.

        Returns
        -------
        map : `~MapBase`
            Downsampled map.

        """
        pass

    @abc.abstractmethod
    def upsample(self, factor):
        """Upsample the spatial dimension of the map by a given factor. 

        Parameters
        ----------
        factor : int
            Upsampling factor.

        Returns
        -------
        map : `~MapBase`
            Upsampled map.

        """
        pass

    def get_by_coords(self, coords, interp=None):
        """Return map values at the given map coordinates.

        Parameters
        ----------
        coords : tuple or `~gammapy.maps.geom.MapCoords`
            `~gammapy.maps.geom.MapCoords` object or tuple of
            coordinate arrays for each dimension of the map.  Tuple
            should be ordered as (lon, lat, x_0, ..., x_n) where x_i
            are coordinates for non-spatial dimensions of the map.

        interp : {None, 'linear', 'nearest'}
            Interpolate data values. None corresponds to 'nearest',
            but might have advantages in performance, because no
            interpolator is set up.

        Returns
        -------
        vals : `~numpy.ndarray`
           Values of pixels in the flattened map.
           np.nan used to flag coords outside of map

        """
        if interp is None:
            idx = self.geom.coord_to_pix(coords)
            return self.get_by_idx(idx)
        else:
            return self.interp_by_coords(coords, interp=interp)

    @abc.abstractmethod
    def get_by_pix(self, pix, interp=None):
        """Return map values at the given pixel coordinates.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel index arrays for each dimension of the map.
            Tuple should be ordered as (I_lon, I_lat, I_0, ..., I_n)
            for WCS maps and (I_hpx, I_0, ..., I_n) for HEALPix maps.
            Pixel indices can be either float or integer type. 

        interp : {None, 'linear', 'nearest'}
            Interpolate data values. None corresponds to 'nearest',
            but might have advantages in performance, because no
            interpolator is set up.

        Returns
        ----------
        vals : `~numpy.ndarray`
           Array of pixel values.
           np.nan used to flag coordinates outside of map

        """
        pass

    @abc.abstractmethod
    def get_by_idx(self, idx):
        """Return map values at the given pixel indices.

        Parameters
        ----------
        idx : tuple
            Tuple of pixel index arrays for each dimension of the map.
            Tuple should be ordered as (I_lon, I_lat, I_0, ..., I_n)
            for WCS maps and (I_hpx, I_0, ..., I_n) for HEALPix maps.

        Returns
        ----------
        vals : `~numpy.ndarray`
           Array of pixel values.
           np.nan used to flag coordinate outside of map

        """
        pass

    @abc.abstractmethod
    def interp_by_coords(self, coords, interp=None):
        """Interpolate map values at the given map coordinates.

        Parameters
        ----------
        coords : tuple or `~gammapy.maps.geom.MapCoords`
            `~gammapy.maps.geom.MapCoords` object or tuple of
            coordinate arrays for each dimension of the map.  Tuple
            should be ordered as (lon, lat, x_0, ..., x_n) where x_i
            are coordinates for non-spatial dimensions of the map.

        interp : {None, 'linear', 'nearest'}
            Interpolate data values. None corresponds to 'nearest',
            but might have advantages in performance, because no
            interpolator is set up.

        Returns
        -------
        vals : `~numpy.ndarray`
           Values of pixels in the flattened map.
           np.nan used to flag coords outside of map

        """
        pass

    def fill_by_coords(self, coords, weights=None):
        """Fill pixels at the given map coordinates with values in `weights`
        vector.

        Parameters
        ----------
        coords : tuple or `~gammapy.maps.geom.MapCoords`
            `~gammapy.maps.geom.MapCoords` object or tuple of
            coordinate arrays for each dimension of the map.  Tuple
            should be ordered as (lon, lat, x_0, ..., x_n) where x_i
            are coordinates for non-spatial dimensions of the map.

        weights : `~numpy.ndarray`
            Weights vector. If None then a unit weight will be assumed
            for each element in `coords`.
        """
        idx = self.geom.coord_to_idx(coords)
        self.fill_by_idx(idx, weights)

    def fill_by_pix(self, pix, weights=None):
        """Fill pixels at the given pixel coordinates with values in `weights`
        vector.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel index arrays for each dimension of the map.
            Tuple should be ordered as (I_lon, I_lat, I_0, ..., I_n)
            for WCS maps and (I_hpx, I_0, ..., I_n) for HEALPix maps.
            Pixel indices can be either float or integer type.  Float
            indices will be rounded to the nearest integer.

        weights : `~numpy.ndarray`
            Weights vector. If None then a unit weight will be assumed
            for each element in `pix`.
        """
        idx = pix_tuple_to_idx(pix)
        return self.fill_by_idx(idx, weights=weights)

    @abc.abstractmethod
    def fill_by_idx(self, idx, weights=None):
        """Fill pixels at the given pixel indices with values in `weights`
        vector.

        Parameters
        ----------
        idx : tuple
            Tuple of pixel index arrays for each dimension of the map.
            Tuple should be ordered as (I_lon, I_lat, I_0, ..., I_n)
            for WCS maps and (I_hpx, I_0, ..., I_n) for HEALPix maps.

        weights : `~numpy.ndarray`
            Weights vector. If None then a unit weight will be assumed
            for each element in `idx`.
        """
        pass

    def set_by_coords(self, coords, vals):
        """Set pixels at the given map coordinates to the values in `vals`
        vector.

        Parameters
        ----------
        coords : tuple or `~gammapy.maps.geom.MapCoords`
            `~gammapy.maps.geom.MapCoords` object or tuple of
            coordinate arrays for each dimension of the map.  Tuple
            should be ordered as (lon, lat, x_0, ..., x_n) where x_i
            are coordinates for non-spatial dimensions of the map.

        vals : `~numpy.ndarray`
            Values vector.  Pixels at `coords` will be set to these values.
        """
        idx = self.geom.coord_to_pix(coords)
        self.set_by_pix(idx, vals)

    def set_by_pix(self, pix, vals):
        """Set pixels at the given pixel coordinates to the values in `vals`
        vector.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel index arrays for each dimension of the map.
            Tuple should be ordered as (I_lon, I_lat, I_0, ..., I_n)
            for WCS maps and (I_hpx, I_0, ..., I_n) for HEALPix maps.
            Pixel indices can be either float or integer type.  Float
            indices will be rounded to the nearest integer.

        vals : `~numpy.ndarray`
            Values vector. Pixels at `pix` will be set to these values.
        """
        idx = pix_tuple_to_idx(pix)
        return self.set_by_idx(idx, vals)

    @abc.abstractmethod
    def set_by_idx(self, idx, vals):
        """Set pixels at the given pixel indices to the values in `vals`
        vector.

        Parameters
        ----------
        idx : tuple
            Tuple of pixel index arrays for each dimension of the map.
            Tuple should be ordered as (I_lon, I_lat, I_0, ..., I_n)
            for WCS maps and (I_hpx, I_0, ..., I_n) for HEALPix maps.

        vals : `~numpy.ndarray`
            Values vector. Pixels at `idx` will be set to these values.
        """
        pass

    def fill_poisson(self, mu):

        pix = self.geom.get_pixels()
        mu = np.random.poisson(mu, len(pix[0]))
        self.fill_by_idx(pix, mu)

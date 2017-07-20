# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
import numpy as np
from astropy.extern import six
from astropy.utils.misc import InheritDocstrings
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
        self._iter = None

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

    def __iter__(self):
        return self

    def __next__(self):

        if self._iter is None:
            self._iter = np.ndenumerate(self.data)

        try:
            return next(self._iter)
        except StopIteration:
            self._iter = None
            raise

    next = __next__

    @abc.abstractmethod
    def sum_over_axes(self):
        """Reduce to a 2D image by summing over non-spatial dimensions."""
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
        pix = self.geom.coord_to_pix(coords)
        self.fill_by_idx(pix, weights)

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

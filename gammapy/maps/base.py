# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
from astropy.extern import six
from astropy.utils.misc import InheritDocstrings

__all__ = [
    'MapBase',
]


class MapMeta(InheritDocstrings, abc.ABCMeta):
    pass


@six.add_metaclass(MapMeta)
class MapBase(object):
    """Abstract map class.  This can represent WCS- or HEALPIX-based maps
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

    @abc.abstractmethod
    def sum_over_axes(self):
        """Reduce to a 2D image by dropping non-spatial dimensions."""
        pass

    @abc.abstractmethod
    def get_by_coords(self, coords, interp=None):
        """Return map values at the given map coordinates.

        Parameters
        ----------
        coords : tuple or `~gammapy.maps.geom.MapCoords`
            `~gammapy.maps.geom.MapCoords` object or tuple of
            coordinate arrays for each dimension of the map.  Tuple
            should be ordered as (lon, lat, x_0, ..., x_n) where x_i
            are coordinates for non-spatial dimensions of the map.

        Returns
        -------
        vals : `~numpy.ndarray`
           Values of pixels in the flattened map.
           np.nan used to flag coords outside of map

        """
        pass

    @abc.abstractmethod
    def get_by_pix(self, pix, interp=None):
        """Return map values at the given pixel coordinates.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel index arrays for each dimension of the map.
            Tuple should be ordered as (I_lon, I_lat, I_0, ..., I_n)
            for WCS maps and (I_hpx, I_0, ..., I_n) for HEALPix maps.

        Returns
        ----------
        vals : `~numpy.ndarray`
           Values of pixels in the flattened map
           np.nan used to flag coords outside of map

        """
        pass

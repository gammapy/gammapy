# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
import numpy as np
from astropy.extern import six
from astropy.coordinates import SkyCoord

__all__ = [
    'MapCoords',
    'MapGeom',
]


def val_to_bin(edges, x):
    """Convert axis coordinates ``x`` to bin indices.

    Returns -1 for values below/above the lower/upper edge.
    """
    ibin = np.digitize(np.array(x, ndmin=1), edges) - 1
    ibin[x > edges[-1]] = -1
    return ibin


def bin_to_val(edges, bins):
    ctr = 0.5 * (edges[1:] + edges[:-1])
    return ctr[bins]


def val_to_pix(edges, x):
    """Convert axis coordinates ``x`` to pixel coordinates.
    """
    return np.interp(x, edges, np.arange(len(center)).astype(float))


class MapCoords(object):
    """Represents a sequence of n-dimensional map coordinates.

    Contains coordinates for 2 spatial dimensions and an arbitrary
    number of additional non-spatial dimensions.

    Parameters
    ----------
    data : tuple of `~numpy.ndarray`
        Data
    """

    def __init__(self, data, coordsys='CEL'):
        self._data = data
        self._coordsys = coordsys

    def __getitem__(self, idx):
        return self._data[idx]

    @property
    def ndim(self):
        return len(self._data)

    @property
    def lon(self):
        return self._data[0]

    @property
    def lat(self):
        return self._data[1]

    @classmethod
    def from_lonlat(cls, lon, lat, *args, **kwargs):
        """Create from vectors of longitude and latitude in degrees."""
        return cls(tuple([lon, lat] + list(args)), **kwargs)

    @classmethod
    def from_skydir(cls, skydir, *args):
        """Create from vector of `~astropy.coordinates.SkyCoord`."""
        if skydir.frame.name in ['icrs', 'fk5']:
            return cls.from_lonlat(skydir.ra.deg, skydir.dec.deg, *args,
                                   coordsys='CEL')
        elif skydir.frame.name in ['galactic']:
            return cls.from_lonlat(skydir.l.deg, skydir.b.deg, *args,
                                   coordsys='GAL')
        else:
            raise Exception('Unrecognized coordinate frame: {}'.format(skydir.frame.name))

    @classmethod
    def from_tuple(cls, coords, **kwargs):
        """Create from tuple of coordinate vectors."""
        if isinstance(coords[0], np.ndarray):
            return cls.from_lonlat(*coords, **kwargs)
        elif isinstance(coords[0], SkyCoord):
            return cls.from_skydir(*coords, **kwargs)
        else:
            raise Exception('Unsupported input type.')

    @classmethod
    def create(cls, data, **kwargs):
        if isinstance(data, cls):
            return data
        elif isinstance(data, tuple):
            return cls.from_tuple(data, **kwargs)
        elif isinstance(data, SkyCoord):
            return cls.from_skydir(data, **kwargs)
        else:
            raise Exception('Unsupported input type.')


@six.add_metaclass(abc.ABCMeta)
class MapGeom(object):
    """Base class for WCS and HEALPIX geometries.
    """

    @abc.abstractmethod
    def coord_to_pix(self, coords):
        """Convert map coordinates to pixel coordinates.

        Parameters
        ----------
        coords : tuple
            Coordinate values in each dimension of the map.  This can
            either be a tuple of numpy arrays or a MapCoords object.
            If passed as a tuple then the ordering should be
            (longitude, latitude, c_0, ..., c_N) where c_i is the
            coordinate vector for axis i.

        Returns
        -------
        pix : tuple
            Tuple of pixel indices in image and band dimensions.

        """
        pass

    @abc.abstractmethod
    def pix_to_coord(self, pix):
        """Convert pixel coordinates to map coordinates.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel coordinates.

        Returns
        -------
        coords : tuple
            Tuple of map coordinates.
        """
        pass

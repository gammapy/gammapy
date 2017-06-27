# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
import numpy as np
from scipy.interpolate import interp1d
from astropy.extern import six
from astropy.coordinates import SkyCoord

__all__ = [
    'MapCoords',
    'MapGeom',
    'MapAxis',
]


def skydir_to_lonlat(skydir, coordsys=None):

    if coordsys in ['CEL', 'C']:
        skydir = skydir.transform_to('icrs')
    elif coordsys in ['GAL', 'G']:
        skydir = skydir.transform_to('galactic')

    if skydir.frame.name in ['icrs', 'fk5']:
        return (skydir.ra.deg, skydir.dec.deg)
    elif skydir.frame.name in ['galactic']:
        return (skydir.l.deg, skydir.b.deg)
    else:
        raise ValueError('Unrecognized SkyCoord frame: {}',
                         skydir.frame.name)


def pix_tuple_to_idx(pix):
    """Convert a tuple of pixel coordinate arrays to a tuple of pixel
    indices.  Pixel coordinates are rounded to the closest integer
    value.

    Returns
    -------
    idx : `~numpy.ndarray`
        Array of pixel indices.

    """
    idx = []
    for i, p in enumerate(pix):
        if np.issubdtype(p.dtype, np.integer):
            idx += [p]
        else:
            idx += [0.5 + p.astype(int)]
    return tuple(idx)


def val_to_bin(edges, x, bounded=False):
    """Convert axis coordinates ``x`` to bin indices.

    Returns -1 for values below/above the lower/upper edge.
    """
    x = np.array(x, ndmin=1)
    ibin = np.digitize(x, edges) - 1

    if bounded:
        ibin[x < edges[0]] = 0
        ibin[x < edges[0]] = len(edges) - 1
    else:
        ibin[x > edges[-1]] = -1
    return ibin


def bin_to_val(edges, bins):
    ctr = 0.5 * (edges[1:] + edges[:-1])
    return ctr[bins]


def coord_to_pix(edges, coord, interp='lin'):
    """Convert grid coordinates to pixel coordinates."""

    if interp == 'log':
        fn = np.log
    elif interp == 'lin':
        def fn(t): return t
    elif interp == 'sqrt':
        fn = np.sqrt
    else:
        raise ValueError('Invalid interp: {}'.format(interp))

    interp_fn = interp1d(fn(edges),
                         np.arange(len(edges)).astype(float),
                         fill_value='extrapolate')

    return interp_fn(fn(coord))


def pix_to_coord(edges, pix, interp='lin'):
    """Convert pixel coordinates to grid coordinates."""

    if interp == 'log':
        fn0 = np.log
        fn1 = np.exp
    elif interp == 'lin':
        def fn0(t): return t
        def fn1(t): return t
    elif interp == 'sqrt':
        fn0 = np.sqrt
        def fn1(t): return np.power(t, 2)
    else:
        raise ValueError('Invalid interp: {}'.format(interp))

    interp_fn = interp1d(np.arange(len(edges)).astype(float),
                 fn0(edges),
                 fill_value='extrapolate')

    return fn1(interp_fn(pix))


def val_to_pix(edges, x):
    """Convert axis coordinates ``x`` to pixel coordinates."""
    return np.interp(x, edges, np.arange(len(edges)).astype(float))


class MapAxis(object):
    """Class representing an axis of a map.  Provides methods for
    converting to/from axis and pixel coordinates.  An axis is defined
    by a sequence of nodes that lie at the center of each bin.  The
    pixel coordinate at each node is equal to its index in the node
    array (0, 1, ..).  Bin edges are offset by 0.5 in pixel
    coordinates from the nodes such that the lower/upper edge of the
    first bin is (-0.5,0.5).

    Parameters
    ----------
    bin_edges : `~numpy.ndarray`
        Array of bin edges.  For a histogram these correspond to the
        lower/upper edges of each bin.

    interp : str
        Interpolation method used to transform between axis and pixel
        coordinates.  Valid options are `log`, `lin`, and `sqrt`.

    """

    def __init__(self, bin_edges, interp='log', name='', quantity_type='integral',
                 nodes=None):
        self._bin_edges = bin_edges
        self._name = name
        self._quantity_type = quantity_type
        self._interp = interp
        self._pix_offset = 0.0

        if nodes is not None:
            self._nodes = nodes
        else:
            self._nodes = bin_edges
            self._pix_offset = 0.5

        pix = np.arange(len(self.edges) - 1, dtype=float)
        self._center = self.pix_to_coord(pix)

    @property
    def name(self):
        """Name of the axis."""
        return self._name

    @property
    def quantity_type(self):
        return self._quantity_type

    @property
    def edges(self):
        """Return array of bin edges."""
        return self._bin_edges

    @property
    def center(self):
        """Return array of bin centers."""
        return self._center

    @property
    def nbin(self):
        return len(self._bin_edges) - 1

    @classmethod
    def from_nodes(cls, nodes, **kwargs):
        """Generate an axis object from a sequence of nodes (bin centers).

        This will create a sequence of bins with edges half-way
        between the node values.  This method should be used to
        construct an axis where the bin center should lie at a
        specific value (e.g. a map of a continuous function).

        Parameters
        ----------
        nodes : `~numpy.ndarray`
            Axis nodes (bin center).

        """
        interp = kwargs.setdefault('interp', 'log')
        pix = np.arange(len(nodes) + 1) - 0.5
        edges = pix_to_coord(nodes, pix, interp=interp)
        return cls(edges, nodes=nodes, **kwargs)

    def set_name(self, name):
        self._name = name

    def pix_to_coord(self, pix):
        pix = pix + self._pix_offset
        return pix_to_coord(self._nodes, pix, interp=self._interp)

    def coord_to_pix(self, coord):
        pix = coord_to_pix(self._nodes, coord, interp=self._interp)
        return pix - self._pix_offset


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
            raise Exception(
                'Unrecognized coordinate frame: {}'.format(skydir.frame.name))

    @classmethod
    def from_tuple(cls, coords, **kwargs):
        """Create from tuple of coordinate vectors."""
        if (isinstance(coords[0], np.ndarray) or
            isinstance(coords[0], list) or
                np.isscalar(coords[0])):
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
    """Base class for WCS and HEALPIX geometries."""

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

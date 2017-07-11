# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
import numpy as np
from astropy.extern import six
from astropy.utils.misc import InheritDocstrings
from astropy import units as u
from astropy.coordinates import SkyCoord

__all__ = [
    'MapCoords',
    'MapGeom',
    'MapAxis',
]


def get_shape(param):
    if param is None:
        return tuple()

    if not isinstance(param, tuple):
        param = [param]
    return max([np.array(p, ndmin=1).shape for p in param])


def coordsys_to_frame(coordsys):
    if coordsys in ['CEL', 'C']:
        return 'icrs'
    elif coordsys in ['GAL', 'G']:
        return 'galactic'
    else:
        raise ValueError('Unrecognized coordinate system: {}',
                         coordsys)


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
        p = np.asarray(p)
        if np.issubdtype(p.dtype, np.integer):
            idx += [p]
        else:
            idx += [(0.5 + p).astype(int)]
    return tuple(idx)


def coord_to_idx(edges, x, bounded=False):
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
    """Convert grid coordinates to pixel coordinates using the chosen
    interpolation scheme."""
    from scipy.interpolate import interp1d

    if interp == 'log':
        fn = np.log
    elif interp == 'lin':
        def fn(t):
            return t
    elif interp == 'sqrt':
        fn = np.sqrt
    else:
        raise ValueError('Invalid interp: {}'.format(interp))

    interp_fn = interp1d(fn(edges),
                         np.arange(len(edges)).astype(float),
                         fill_value='extrapolate')

    return interp_fn(fn(coord))


def pix_to_coord(edges, pix, interp='lin'):
    """Convert pixel coordinates to grid coordinates using the chosen
    interpolation scheme."""
    from scipy.interpolate import interp1d

    if interp == 'log':
        fn0 = np.log
        fn1 = np.exp
    elif interp == 'lin':
        def fn0(t):
            return t

        def fn1(t):
            return t
    elif interp == 'sqrt':
        fn0 = np.sqrt

        def fn1(t):
            return np.power(t, 2)
    else:
        raise ValueError('Invalid interp: {}'.format(interp))

    interp_fn = interp1d(np.arange(len(edges)).astype(float),
                         fn0(edges),
                         fill_value='extrapolate')

    return fn1(interp_fn(pix))


class MapAxis(object):
    """Class representing an axis of a map.  Provides methods for
    transforming to/from axis and pixel coordinates.  An axis is
    defined by a sequence of node values that lie at the center of
    each bin.  The pixel coordinate at each node is equal to its index
    in the node array (0, 1, ..).  Bin edges are offset by 0.5 in
    pixel coordinates from the nodes such that the lower/upper edge of
    the first bin is (-0.5,0.5).

    Parameters
    ----------
    nodes : `~numpy.ndarray`
        Array of node values.  These will be interpreted as either bin
        edges or centers.
    interp : str
        Interpolation method used to transform between axis and pixel
        coordinates.  Valid options are `log`, `lin`, and `sqrt`.
    unit : str
        String specifying the data units.
    """

    # TODO: Add methods to faciliate FITS I/O.
    # TODO: Cache an interpolation object?

    def __init__(self, nodes, interp='lin', name='', quantity_type='integral',
                 node_type='edge', unit=''):
        self._name = name
        self._quantity_type = quantity_type
        self._interp = interp
        self._nodes = nodes
        self._node_type = node_type
        self._unit = u.Unit(unit)

        # Set pixel coordinate of first node
        if node_type == 'edge':
            self._pix_offset = -0.5
            nbin = len(nodes) - 1
        elif node_type == 'center':
            self._pix_offset = 0.0
            nbin = len(nodes)
        else:
            raise ValueError('Invalid node type: {}'.format(node_type))

        pix = np.arange(nbin, dtype=float)
        self._center = self.pix_to_coord(pix)
        pix = np.arange(nbin + 1, dtype=float) - 0.5
        self._bin_edges = self.pix_to_coord(pix)

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
        """Return number of bins."""
        return len(self._bin_edges) - 1

    @property
    def unit(self):
        """Return coordinate axis unit."""
        return self._unit

    @classmethod
    def from_bounds(cls, lo_bnd, hi_bnd, nbin, **kwargs):
        """Generate an axis object from a lower/upper bound and number of
        bins.

        Parameters
        ----------
        lo_bnd : float
            Lower bound of first axis bin.
        hi_bnd : float
            Upper bound of last axis bin.
        nbin : int
            Number of bins.
        interp : str
            Interpolation method used to transform between axis and pixel
            coordinates.  Valid options are `log`, `lin`, and `sqrt`.
        """

        interp = kwargs.setdefault('interp', 'lin')

        if interp == 'lin':
            nodes = np.linspace(lo_bnd, hi_bnd, nbin + 1)
        elif interp == 'log':
            nodes = np.exp(np.linspace(np.log(lo_bnd),
                                       np.log(hi_bnd), nbin + 1))
        elif interp == 'sqrt':
            nodes = np.linspace(lo_bnd ** 0.5,
                                hi_bnd ** 0.5, nbin + 1) ** 2.0
        else:
            raise ValueError('Invalid interp: {}'.format(interp))

        return cls(nodes, node_type='edge', **kwargs)

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
        interp : str
            Interpolation method used to transform between axis and pixel
            coordinates.  Valid options are `log`, `lin`, and `sqrt`.
        """
        nodes = np.array(nodes, ndmin=1)
        if len(nodes) < 1:
            raise ValueError('Nodes array must have at least one element.')

        return cls(nodes, node_type='center', **kwargs)

    @classmethod
    def from_edges(cls, edges, **kwargs):
        """Generate an axis object from a sequence of bin edges.

        This method should be used to construct an axis where the bin
        edges should lie at specific values (e.g. a histogram).  The
        number of bins will be one less than the number of edges.

        Parameters
        ----------
        edges : `~numpy.ndarray`
            Axis bin edges.
        interp : str
            Interpolation method used to transform between axis and pixel
            coordinates.  Valid options are `log`, `lin`, and `sqrt`.
        """
        if len(edges) < 2:
            raise ValueError('Edges array must have at least two elements.')

        return cls(edges, node_type='edge', **kwargs)

    def set_name(self, name):
        self._name = name

    def pix_to_coord(self, pix):
        """Transform from pixel to axis coordinates.

        Parameters
        ----------
        pix : `~numpy.ndarray`
            Array of pixel coordinate values.

        Returns
        -------
        coord : `~numpy.ndarray`
            Array of axis coordinate values.
        """
        pix = pix - self._pix_offset
        return pix_to_coord(self._nodes, pix, interp=self._interp)

    def coord_to_pix(self, coord):
        """Transform from axis to pixel coordinates.

        Parameters
        ----------
        coord : `~numpy.ndarray`
            Array of axis coordinate values.

        Returns
        -------
        pix : `~numpy.ndarray`
            Array of pixel coordinate values.
        """
        pix = coord_to_pix(self._nodes, coord, interp=self._interp)
        return np.array(pix + self._pix_offset, ndmin=1)

    def coord_to_idx(self, coord, bounded=False):
        """Transform from axis coordinate to bin index.

        Parameters
        ----------
        coord : `~numpy.ndarray`
            Array of axis coordinate values.
        bounded : bool


        Returns
        -------
        idx : `~numpy.ndarray`
            Array of bin indices.
        """
        return coord_to_idx(self.edges, coord, bounded)

    def slice(self, idx):
        """Create a new axis object by extracting a slice from this axis.

        Parameters
        ----------
        idx : slice
            Slice object selecting a subselection of the axis.

        Returns
        -------
        axis : `~MapAxis`
            Sliced axis objected.
        """
        center = self.center[idx]
        idx = self.coord_to_idx(center)
        # For edge nodes we need to keep N+1 nodes
        if self._node_type == 'edge':
            idx = tuple(list(idx) + [1 + idx[-1]])
        nodes = self._nodes[(idx,)]
        return MapAxis(nodes, interp=self._interp, name=self._name,
                       quantity_type=self._quantity_type,
                       node_type=self._node_type, unit=self._unit)


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
        elif isinstance(data, tuple) or isinstance(data, list):
            return cls.from_tuple(data, **kwargs)
        elif isinstance(data, SkyCoord):
            return cls.from_skydir(data, **kwargs)
        else:
            raise Exception('Unsupported input type.')


class MapGeomMeta(InheritDocstrings, abc.ABCMeta):
    pass


@six.add_metaclass(MapGeomMeta)
class MapGeom(object):
    """Base class for WCS and HEALPix geometries."""

    @abc.abstractproperty
    def center_coord(self):
        pass

    @abc.abstractproperty
    def center_pix(self):
        pass

    @abc.abstractproperty
    def center_skydir(self):
        pass

    @abc.abstractmethod
    def get_pixels(self):
        """Get pixel indices for all pixels in this geometry.

        Returns
        -------
        pix : tuple
            Tuple of pixel index vectors with one element for each
            dimension.
        """
        pass

    @abc.abstractmethod
    def get_coords(self):
        """Get the coordinates of all the pixels in this geometry.

        Returns
        -------
        coords : tuple
            Tuple of coordinate vectors with one element for each
            dimension.
        """
        pass

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
            Tuple of pixel coordinates in image and band dimensions.
        """
        pass

    def coord_to_idx(self, coords):
        """Convert map coordinates to pixel indices.

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
            Elements set to -1 correspond to coordinates outside the
            map.
        """
        pix = self.coord_to_pix(coords)
        return self.pix_to_idx(pix)

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

    @abc.abstractmethod
    def pix_to_idx(self, pix):
        """Convert pixel coordinates to pixel indices.  Returns -1 for pixel
        coordinates that lie outside of the map.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel coordinates.

        Returns
        -------
        idx : tuple
            Tuple of pixel indices.
        """
        pass

    @abc.abstractmethod
    def contains(self, coords):
        """
        Check if a given coordinate is contained in the map.

        Parameters
        ----------
        coords : tuple
            Tuple of map coordinates.

        Returns
        -------
        containment : `~np.ndarray`
            Bool array
        """
        pass

    @abc.abstractmethod
    def to_slice(self, slices, drop_axes=True):
        """Create a new geometry by cutting in the non-spatial dimensions of
        this geometry.

        Parameters
        ----------
        slices : tuple
            Tuple of integers or `slice` objects.  Contains one
            element for each non-spatial dimension.

        drop_axes : bool
            Drop axes for which the slice reduces the size of that
            dimension to one.

        Returns
        -------
        geom : `~MapGeom`
            Sliced geometry.
        """
        pass

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import copy
import inspect
import logging
from collections.abc import Sequence
import numpy as np
import scipy.interpolate
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Column, Table, hstack
from astropy.utils import lazyproperty
from gammapy.utils.interpolation import interpolation_scale
from .utils import INVALID_INDEX, edges_from_lo_hi, find_bands_hdu, find_hdu

__all__ = ["MapCoord", "Geom", "MapAxis", "MapAxes"]

log = logging.getLogger(__name__)


def get_shape(param):
    if param is None:
        return tuple()

    if not isinstance(param, tuple):
        param = [param]

    return max([np.array(p, ndmin=1).shape for p in param])


def skycoord_to_lonlat(skycoord, frame=None):
    """Convert SkyCoord to lon, lat, frame.

    Returns
    -------
    lon : `~numpy.ndarray`
        Longitude in degrees.
    lat : `~numpy.ndarray`
        Latitude in degrees.
    """
    if frame:
        skycoord = skycoord.transform_to(frame)

    return skycoord.data.lon.deg, skycoord.data.lat.deg, skycoord.frame.name


def pix_tuple_to_idx(pix):
    """Convert a tuple of pixel coordinate arrays to a tuple of pixel indices.

    Pixel coordinates are rounded to the closest integer value.

    Parameters
    ----------
    pix : tuple
        Tuple of pixel coordinates with one element for each dimension

    Returns
    -------
    idx : `~numpy.ndarray`
        Array of pixel indices
    """
    idx = []
    for p in pix:
        p = np.array(p, ndmin=1)
        if np.issubdtype(p.dtype, np.integer):
            idx += [p]
        else:
            p_idx = np.rint(p).astype(int)
            p_idx[~np.isfinite(p)] = INVALID_INDEX.int
            idx += [p_idx]

    return tuple(idx)


def coord_to_pix(edges, coord, interp="lin"):
    """Convert axis to pixel coordinates for given interpolation scheme."""
    scale = interpolation_scale(interp)

    interp_fn = scipy.interpolate.interp1d(
        scale(edges), np.arange(len(edges), dtype=float), fill_value="extrapolate"
    )

    return interp_fn(scale(coord))


def pix_to_coord(edges, pix, interp="lin"):
    """Convert pixel to grid coordinates for given interpolation scheme."""
    scale = interpolation_scale(interp)

    interp_fn = scipy.interpolate.interp1d(
        np.arange(len(edges), dtype=float), scale(edges), fill_value="extrapolate"
    )

    return scale.inverse(interp_fn(pix))


class MapAxes(Sequence):
    """MapAxis container class.

    Parameters
    ----------
    axes : list of `MapAxis`
        List of map axis objects.
    """

    def __init__(self, axes):
        unique_names = []
        for ax in axes:
            if ax.name in unique_names:
                raise (ValueError("Axis names must be unique"))
            unique_names.append(ax.name)

        self._axes = axes

    @property
    def shape(self):
        """Shape of the axes"""
        return tuple([ax.nbin for ax in self])

    @property
    def names(self):
        """Names of the axes"""
        return [ax.name for ax in self]

    def index(self, axis_name):
        """Get index in list"""
        return self.names.index(axis_name)

    def index_data(self, axis_name):
        """Get data index of the axes

        Parameters
        ----------
        axis_name : str
            Name of the axis.

        Returns
        -------
        idx : int
            Data index
        """
        idx = self.names.index(axis_name)
        return len(self) - idx - 1

    def __len__(self):
        return len(self._axes)

    def __add__(self, other):
        return self.__class__(list(self) + list(other))

    def upsample(self, factor, axis_name):
        """Upsample axis by a given factor

        Parameters
        ----------
        factor : int
            Upsampling factor.
        axis_name : str
            Axis to upsample.

        Returns
        -------
        axes : `MapAxes`
            Map axes
        """
        axes = []

        for ax in self:
            if ax.name == axis_name:
                ax = ax.upsample(factor=factor)

            axes.append(ax.copy())

        return self.__class__(axes=axes)

    def resample(self, axis):
        """Resample axis binning.

        This method groups the existing bins into a new binning.

        Parameters
        ----------
        axis : `MapAxis`
            New map axis.

        Returns
        -------
        axes : `MapAxes`
            Axes object with resampled axis.
        """
        axis_self = self[axis.name]
        groups = axis_self.group_table(axis.edges)

        # Keep only normal bins
        groups = groups[groups["bin_type"] == "normal   "]

        edges = edges_from_lo_hi(
            groups[axis.name + "_min"].quantity, groups[axis.name + "_max"].quantity,
        )

        axis_resampled = MapAxis.from_edges(
            edges=edges, interp=axis.interp, name=axis.name
        )

        axes = []
        for ax in self:
            if ax.name == axis.name:
                axes.append(axis_resampled)
            else:
                axes.append(ax.copy())

        return self.__class__(axes=axes)

    def downsample(self, factor, axis_name):
        """Downsample axis by a given factor

        Parameters
        ----------
        factor : int
            Upsampling factor.
        axis_name : str
            Axis to upsample.

        Returns
        -------
        axes : `MapAxes`
            Map axes

        """
        axes = []

        for ax in self:
            if ax.name == axis_name:
                ax = ax.downsample(factor=factor)

            axes.append(ax.copy())

        return self.__class__(axes=axes)

    def squash(self, axis_name):
        """Squash axis.

        Parameters
        ----------
        axis_name : str
            Axis to squash.

        Returns
        -------
        axes : `MapAxes`
            Axes with squashed axis.
        """
        axes = []

        for ax in self:
            if ax.name == axis_name:
                ax = ax.squash()
            axes.append(ax.copy())

        return self.__class__(axes=axes)

    def drop(self, axis_name):
        """Drop an axis.

        Parameters
        ----------
        axis_name : str
            Name of the axis to remove.

        Returns
        -------
        axes : `MapAxes`
            Axes with squashed axis.
        """
        axes = []
        for ax in self:
            if ax.name == axis_name:
                continue
            axes.append(ax.copy())

        return self.__class__(axes=axes)

    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):
            return self._axes[idx]
        elif isinstance(idx, str):
            for ax in self._axes:
                if ax.name == idx:
                    return ax
            raise KeyError(f"No axes: {idx!r}")
        else:
            raise TypeError(f"Invalid type: {type(idx)!r}")

    def coord_to_idx(self, coord, clip=True):
        """Transform from axis to pixel indices.

        Parameters
        ----------
        coord : dict of `~numpy.ndarray` or `MapCoord`
            Array of axis coordinate values.

        Returns
        -------
        pix : tuple of `~numpy.ndarray`
            Array of pixel indices values.
        """
        return tuple([ax.coord_to_idx(coord[ax.name], clip=clip) for ax in self])

    def coord_to_pix(self, coord):
        """Transform from axis to pixel coordinates.

        Parameters
        ----------
        coord : dict of `~numpy.ndarray`
            Array of axis coordinate values.

        Returns
        -------
        pix : tuple of `~numpy.ndarray`
            Array of pixel coordinate values.
        """
        return tuple([ax.coord_to_pix(coord[ax.name]) for ax in self])

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
        return tuple([ax.pix_to_coord(p) for ax, p in zip(self, pix)])

    def slice_by_idx(self, slices):
        """Create a new geometry by slicing the non-spatial axes.

        Parameters
        ----------
        slices : dict
            Dict of axes names and integers or `slice` object pairs. Contains one
            element for each non-spatial dimension. For integer indexing the
            corresponding axes is dropped from the map. Axes not specified in the
            dict are kept unchanged.

        Returns
        -------
        geom : `~Geom`
            Sliced geometry.
        """
        axes = []
        for ax in self:
            ax_slice = slices.get(ax.name, slice(None))

            # in the case where isinstance(ax_slice, int) the axes is dropped
            if isinstance(ax_slice, slice):
                ax_sliced = ax.slice(ax_slice)
                axes.append(ax_sliced.copy())

        return self.__class__(axes=axes)

    def to_header(self, header=None):
        """Convert axes to FITS header

        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            If a header is provided it is extended with the information.
            Otherwise a new one is created.

        Returns
        -------
        header : `~astropy.io.fits.Header`
            FITS header.
        """
        if header is None:
            header = fits.Header()

        for idx, ax in enumerate(self, start=1):
            key = f"AXCOLS{idx}"
            name = ax.name.upper()
            if ax.name == "energy" and ax.node_type == "edges":
                header[key] = "E_MIN,E_MAX"
            elif ax.name == "energy" and ax.node_type == "center":
                header[key] = "ENERGY"
            elif ax.node_type == "edges":
                header[key] = f"{name}_MIN,{name}_MAX"
            elif ax.node_type == "center":
                header[key] = name
            else:
                raise ValueError(f"Invalid node type {ax.node_type!r}")

            key_interp = f"INTERP{idx}"
            header[key_interp] = ax.interp
        return header

    def to_table(self, format=None):
        """Convert axes to table

        Parameters
        ----------
        format : {"gadf-dl3"}
            Format to use.

        Returns
        -------
        table : `~astropy.table.Table`
            Table with axis data
        """

        tables = []

        for ax in self:
            tables.append(ax.to_table(format=format))

        return hstack(tables)

    def to_table_hdu(self, format=None, prefix=None):
        """Make FITS table columns for map axes.

        Parameters
        ----------
        format : {"gadf", "fgst-ccube", "fgst-template"}
            Format to use.
        prefix : str
            HDU name prefix to use

        Returns
        -------
        hdu : `~astropy.io.fits.BinTableHDU`
            Bin table HDU.
        """
        # FIXME: Check whether convention is compatible with
        #  dimensionality of geometry and simplify!!!

        if format == "fgst-ccube":
            hdu = "EBOUNDS"
        elif format == "fgst-template":
            hdu = "ENERGIES"
        elif format == "gadf" or format is None:
            if prefix:
                hdu = f"{prefix}_BANDS"
            else:
                hdu = "BANDS"
        else:
            raise ValueError(f"Unknown format {format}")

        size = np.prod([ax.nbin for ax in self])
        chan = np.arange(0, size)
        cols = [fits.Column("CHANNEL", "I", array=chan)]

        axes_ctr = np.meshgrid(*[ax.center for ax in self])
        axes_min = np.meshgrid(*[ax.edges[:-1] for ax in self])
        axes_max = np.meshgrid(*[ax.edges[1:] for ax in self])

        for idx, ax in enumerate(self):
            name = ax.name.upper()

            if name == "ENERGY":
                colnames = ["ENERGY", "E_MIN", "E_MAX"]
            else:
                colnames = [name, name + "_MIN", name + "_MAX"]

            for colname, v in zip(colnames, [axes_ctr, axes_min, axes_max]):
                array = np.ravel(v[idx])
                unit = ax.unit.to_string("fits")
                cols.append(fits.Column(colname, "E", array=array, unit=unit))

        header = self.to_header()
        return fits.BinTableHDU.from_columns(cols, name=hdu, header=header)

    @classmethod
    def from_table_hdu(cls, hdu, format=None):
        """Create MapAxes from BinTableHDU
        """
        if hdu is None:
            return cls([])

        if format in ["fgst-ccube", "fgst-template", "fgst-bexpcube"]:
            axes = [MapAxis.from_table_hdu(hdu, format=format)]
        else:
            axes = []

            for idx in range(5):
                try:
                    axis = MapAxis.from_table_hdu(hdu, format="gadf", idx=idx)
                    axes.append(axis)
                except AttributeError:
                    continue

        return cls(axes)

    @classmethod
    def from_default(cls, axes):
        """Make a sequence of `~MapAxis` objects."""
        if axes is None:
            return cls([])

        axes_out = []
        for idx, ax in enumerate(axes):
            if isinstance(ax, np.ndarray):
                ax = MapAxis(ax)

            if ax.name == "":
                ax.name = f"axis{idx}"

            axes_out.append(ax)

        return cls(axes_out)


class MapAxis:
    """Class representing an axis of a map.

    Provides methods for
    transforming to/from axis and pixel coordinates.  An axis is
    defined by a sequence of node values that lie at the center of
    each bin.  The pixel coordinate at each node is equal to its index
    in the node array (0, 1, ..).  Bin edges are offset by 0.5 in
    pixel coordinates from the nodes such that the lower/upper edge of
    the first bin is (-0.5,0.5).

    Parameters
    ----------
    nodes : `~numpy.ndarray` or `~astropy.units.Quantity`
        Array of node values.  These will be interpreted as either bin
        edges or centers according to ``node_type``.
    interp : str
        Interpolation method used to transform between axis and pixel
        coordinates.  Valid options are 'log', 'lin', and 'sqrt'.
    name : str
        Axis name
    node_type : str
        Flag indicating whether coordinate nodes correspond to pixel
        edges (node_type = 'edge') or pixel centers (node_type =
        'center').  'center' should be used where the map values are
        defined at a specific coordinate (e.g. differential
        quantities). 'edge' should be used where map values are
        defined by an integral over coordinate intervals (e.g. a
        counts histogram).
    unit : str
        String specifying the data units.
    """

    # TODO: Add methods to faciliate FITS I/O.
    # TODO: Cache an interpolation object?
    def __init__(self, nodes, interp="lin", name="", node_type="edges", unit=""):
        self._name = name

        if len(nodes) != len(np.unique(nodes)):
            raise ValueError("MapAxis: node values must be unique")
        if ~(np.all(nodes == np.sort(nodes)) or np.all(nodes[::-1] == np.sort(nodes))):
            raise ValueError("MapAxis: node values must be sorted")

        if len(nodes) == 1 and node_type == "center":
            raise ValueError("Single bins can only be used with node-type 'edges'")

        if isinstance(nodes, u.Quantity):
            unit = nodes.unit if nodes.unit is not None else ""
            nodes = nodes.value
        else:
            nodes = np.array(nodes)

        self._unit = u.Unit(unit)
        self._nodes = nodes.astype(float)
        self._node_type = node_type
        self._interp = interp

        if (self._nodes < 0).any() and interp != "lin":
            raise ValueError(
                f"Interpolation scaling {interp!r} only support for positive node values."
            )

        # Set pixel coordinate of first node
        if node_type == "edges":
            self._pix_offset = -0.5
            nbin = len(nodes) - 1
        elif node_type == "center":
            self._pix_offset = 0.0
            nbin = len(nodes)
        else:
            raise ValueError(f"Invalid node type: {node_type!r}")

        self._nbin = nbin

    def is_aligned(self, other, atol=2e-2):
        """Check if other map axis is aligned.

        Two axes are aligned if their center coordinate values map to integers
        on the other axes as well and if the interpolation modes are equivalent.

        Parameters
        ----------
        other : `MapAxis`
            Other map axis.
        atol : float
            Absolute numerical tolerance for the comparison measured in bins.

        Returns
        -------
        aligned : bool
            Whether the axes are aligned
        """
        pix = self.coord_to_pix(other.center)
        pix_other = other.coord_to_pix(self.center)
        pix_all = np.append(pix, pix_other)
        aligned = np.allclose(np.round(pix_all) - pix_all, 0, atol=atol)
        return aligned and self.interp == other.interp

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        # TODO: implement an allclose method for MapAxis and call it here
        if self.edges.shape != other.edges.shape:
            return False
        if self.unit.is_equivalent(other.unit) is False:
            return False
        return (
            np.allclose(
                self.edges.to(other.unit).value, other.edges.value, atol=1e-6, rtol=1e-6
            )
            and self._node_type == other._node_type
            and self._interp == other._interp
            and self.name.upper() == other.name.upper()
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return id(self)

    @property
    def is_energy_axis(self):
        return self.name in ["energy", "energy_true"]

    @property
    def interp(self):
        """Interpolation scale of the axis."""
        return self._interp

    @property
    def name(self):
        """Name of the axis."""
        return self._name

    @name.setter
    def name(self, value):
        """Name of the axis."""
        self._name = value

    @lazyproperty
    def edges(self):
        """Return array of bin edges."""
        pix = np.arange(self.nbin + 1, dtype=float) - 0.5
        return u.Quantity(self.pix_to_coord(pix), self._unit, copy=False)

    @lazyproperty
    def center(self):
        """Return array of bin centers."""
        pix = np.arange(self.nbin, dtype=float)
        return u.Quantity(self.pix_to_coord(pix), self._unit, copy=False)

    @lazyproperty
    def bin_width(self):
        """Array of bin widths."""
        return np.diff(self.edges)

    @property
    def nbin(self):
        """Return number of bins."""
        return self._nbin

    @property
    def nbin_per_decade(self):
        """Return number of bins."""
        if self.interp != "log":
            raise ValueError("Bins per decade can only be computed for log-spaced axes")

        if self.node_type == "edges":
            values = self.edges
        else:
            values = self.center

        ndecades = np.log10(values.max() / values.min())
        return (self._nbin / ndecades).value

    @property
    def node_type(self):
        """Return node type ('center' or 'edge')."""
        return self._node_type

    @property
    def unit(self):
        """Return coordinate axis unit."""
        return self._unit

    @classmethod
    def from_bounds(cls, lo_bnd, hi_bnd, nbin, **kwargs):
        """Generate an axis object from a lower/upper bound and number of bins.

        If node_type = 'edge' then bounds correspond to the
        lower and upper bound of the first and last bin.  If node_type
        = 'center' then bounds correspond to the centers of the first
        and last bin.

        Parameters
        ----------
        lo_bnd : float
            Lower bound of first axis bin.
        hi_bnd : float
            Upper bound of last axis bin.
        nbin : int
            Number of bins.
        interp : {'lin', 'log', 'sqrt'}
            Interpolation method used to transform between axis and pixel
            coordinates.  Default: 'lin'.
        """
        nbin = int(nbin)
        interp = kwargs.setdefault("interp", "lin")
        node_type = kwargs.setdefault("node_type", "edges")

        if node_type == "edges":
            nnode = nbin + 1
        elif node_type == "center":
            nnode = nbin
        else:
            raise ValueError(f"Invalid node type: {node_type!r}")

        if interp == "lin":
            nodes = np.linspace(lo_bnd, hi_bnd, nnode)
        elif interp == "log":
            nodes = np.exp(np.linspace(np.log(lo_bnd), np.log(hi_bnd), nnode))
        elif interp == "sqrt":
            nodes = np.linspace(lo_bnd ** 0.5, hi_bnd ** 0.5, nnode) ** 2.0
        else:
            raise ValueError(f"Invalid interp: {interp}")

        return cls(nodes, **kwargs)

    @classmethod
    def from_energy_edges(cls, energy_edges, unit=None, name=None, interp="log"):
        """Make an energy axis from adjacent edges.

        Parameters
        ----------
        energy_edges : `~astropy.units.Quantity`, float
            Energy edges
        unit : `~astropy.units.Unit`
            Energy unit
        name : str
            Name of the energy axis, either 'energy' or 'energy_true'
        interp: str
            interpolation mode. Default is 'log'.

        Returns
        -------
        axis : `MapAxis`
            Axis with name "energy" and interp "log".
        """
        energy_edges = u.Quantity(energy_edges, unit)

        if unit is None:
            unit = energy_edges.unit
            energy_edges = energy_edges.to(unit)

        if name is None:
            name = "energy"

        if name not in ["energy", "energy_true"]:
            raise ValueError("Energy axis can only be named 'energy' or 'energy_true'")

        return cls.from_edges(energy_edges, unit=unit, interp=interp, name=name)

    @classmethod
    def from_energy_bounds(
        cls,
        energy_min,
        energy_max,
        nbin,
        unit=None,
        per_decade=False,
        name=None,
        node_type="edges",
    ):
        """Make an energy axis.

        Used frequently also to make energy grids, by making
        the axis, and then using ``axis.center`` or ``axis.edges``.

        Parameters
        ----------
        energy_min, energy_max : `~astropy.units.Quantity`, float
            Energy range
        nbin : int
            Number of bins
        unit : `~astropy.units.Unit`
            Energy unit
        per_decade : bool
            Whether `nbin` is given per decade.
        name : str
            Name of the energy axis, either 'energy' or 'energy_true'

        Returns
        -------
        axis : `MapAxis`
            Axis with name "energy" and interp "log".
        """
        energy_min = u.Quantity(energy_min, unit)
        energy_max = u.Quantity(energy_max, unit)

        if unit is None:
            unit = energy_max.unit
            energy_min = energy_min.to(unit)

        if per_decade:
            nbin = np.ceil(np.log10(energy_max / energy_min).value * nbin)

        if name is None:
            name = "energy"

        if name not in ["energy", "energy_true"]:
            raise ValueError("Energy axis can only be named 'energy' or 'energy_true'")

        return cls.from_bounds(
            energy_min.value,
            energy_max.value,
            nbin=nbin,
            unit=unit,
            interp="log",
            name=name,
            node_type=node_type,
        )

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
        interp : {'lin', 'log', 'sqrt'}
            Interpolation method used to transform between axis and pixel
            coordinates.  Default: 'lin'.
        """
        if len(nodes) < 1:
            raise ValueError("Nodes array must have at least one element.")

        return cls(nodes, node_type="center", **kwargs)

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
        interp : {'lin', 'log', 'sqrt'}
            Interpolation method used to transform between axis and pixel
            coordinates.  Default: 'lin'.
        """
        if len(edges) < 2:
            raise ValueError("Edges array must have at least two elements.")

        return cls(edges, node_type="edges", **kwargs)

    def append(self, axis):
        """Append another map axis to this axis

        Name, interp type and node type must agree between the axes. If the node
        type is "edges", the edges must be contiguous and non-overlapping.

        Parameters
        ----------
        axis : `MapAxis`
            Axis to append.

        Returns
        -------
        axis : `MapAxis`
            Appended axis
        """
        if self.node_type != axis.node_type:
            raise ValueError(
                f"Node type must agree, got {self.node_type} and {axis.node_type}"
            )

        if self.name != axis.name:
            raise ValueError(f"Names must agree, got {self.name} and {axis.name} ")

        if self.interp != axis.interp:
            raise ValueError(
                f"Interp type must agree, got {self.interp} and {axis.interp}"
            )

        if self.node_type == "edges":
            edges = np.append(self.edges, axis.edges[1:])
            return self.from_edges(edges=edges, interp=self.interp, name=self.name)
        else:
            nodes = np.append(self.center, axis.center)
            return self.from_nodes(nodes=nodes, interp=self.interp, name=self.name)

    @classmethod
    def from_stack(cls, axes):
        """Create a map axis by merging a list of other map axes.

        If the node type is "edges" the bin edges in the provided axes must be
        contiguous and non-overlapping.

        Parameters
        ----------
        axes : list of `MapAxis`
            List of map axis to merge.

        Returns
        -------
        axis : `MapAxis`
            Merged axis
        """
        ax_stacked = axes[0]

        for ax in axes[1:]:
            ax_stacked = ax_stacked.append(ax)

        return ax_stacked

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
        values = pix_to_coord(self._nodes, pix, interp=self._interp)
        return u.Quantity(values, unit=self.unit, copy=False)

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
        coord = u.Quantity(coord, self.unit, copy=False).value
        pix = coord_to_pix(self._nodes, coord, interp=self._interp)
        return np.array(pix + self._pix_offset, ndmin=1)

    def coord_to_idx(self, coord, clip=False):
        """Transform from axis coordinate to bin index.

        Parameters
        ----------
        coord : `~numpy.ndarray`
            Array of axis coordinate values.
        clip : bool
            Choose whether to clip the index to the valid range of the
            axis.  If false then indices for values outside the axis
            range will be set -1.

        Returns
        -------
        idx : `~numpy.ndarray`
            Array of bin indices.
        """
        coord = u.Quantity(coord, self.unit, copy=False, ndmin=1).value
        edges = self.edges.value
        idx = np.digitize(coord, edges) - 1

        if clip:
            idx = np.clip(idx, 0, self.nbin - 1)
        else:
            with np.errstate(invalid="ignore"):
                idx[coord > edges[-1]] = INVALID_INDEX.int

        idx[~np.isfinite(coord)] = INVALID_INDEX.int

        return idx

    def slice(self, idx):
        """Create a new axis object by extracting a slice from this axis.

        Parameters
        ----------
        idx : slice
            Slice object selecting a subselection of the axis.

        Returns
        -------
        axis : `~MapAxis`
            Sliced axis object.
        """
        center = self.center[idx].value
        idx = self.coord_to_idx(center)
        # For edge nodes we need to keep N+1 nodes
        if self._node_type == "edges":
            idx = tuple(list(idx) + [1 + idx[-1]])

        nodes = self._nodes[(idx,)]
        return MapAxis(
            nodes,
            interp=self._interp,
            name=self._name,
            node_type=self._node_type,
            unit=self._unit,
        )

    def squash(self):
        """Create a new axis object by squashing the axis into one bin.

        Returns
        -------
        axis : `~MapAxis`
            Sliced axis object.
        """
        # TODO: Decide on handling node_type=center
        # See https://github.com/gammapy/gammapy/issues/1952
        return MapAxis.from_bounds(
            lo_bnd=self.edges[0].value,
            hi_bnd=self.edges[-1].value,
            nbin=1,
            interp=self._interp,
            name=self._name,
            unit=self._unit,
        )

    def __repr__(self):
        str_ = self.__class__.__name__
        str_ += "\n\n"
        fmt = "\t{:<10s} : {:<10s}\n"
        str_ += fmt.format("name", self.name)
        str_ += fmt.format("unit", "{!r}".format(str(self.unit)))
        str_ += fmt.format("nbins", str(self.nbin))
        str_ += fmt.format("node type", self.node_type)
        vals = self.edges if self.node_type == "edges" else self.center
        str_ += fmt.format(f"{self.node_type} min", "{:.1e}".format(vals.min()))
        str_ += fmt.format(f"{self.node_type} max", "{:.1e}".format(vals.max()))
        str_ += fmt.format("interp", self._interp)
        return str_

    def _init_copy(self, **kwargs):
        """Init map axis instance by copying missing init arguments from self.
        """
        argnames = inspect.getfullargspec(self.__init__).args
        argnames.remove("self")

        for arg in argnames:
            value = getattr(self, "_" + arg)
            kwargs.setdefault(arg, copy.deepcopy(value))

        return self.__class__(**kwargs)

    def copy(self, **kwargs):
        """Copy `MapAxis` instance and overwrite given attributes.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to overwrite in the map axis constructor.

        Returns
        -------
        copy : `MapAxis`
            Copied map axis.
        """
        return self._init_copy(**kwargs)

    def round(self, coord, clip=False):
        """Round coord to nearest axis edge.

        Parameters
        ----------
        coord : `~astropy.units.Quantity`
            Coordinates
        clip : bool
            Choose whether to clip indices to the valid range of the axis.

        Returns
        -------
        coord : `~astropy.units.Quantity`
            Rounded coordinates
        """
        edges_pix = self.coord_to_pix(coord)

        if clip:
            edges_pix = np.clip(edges_pix, -0.5, self.nbin - 0.5)

        edges_idx = np.round(edges_pix + 0.5) - 0.5
        return self.pix_to_coord(edges_idx)

    def group_table(self, edges):
        """Compute bin groups table for the map axis, given coarser bin edges.

        Parameters
        ----------
        edges : `~astropy.units.Quantity`
            Group bin edges.

        Returns
        -------
        groups : `~astropy.table.Table`
            Map axis group table.
        """
        # TODO: try to simplify this code
        if not self.node_type == "edges":
            raise ValueError("Only edge based map axis can be grouped")

        edges_pix = self.coord_to_pix(edges)
        edges_pix = np.clip(edges_pix, -0.5, self.nbin - 0.5)
        edges_idx = np.round(edges_pix + 0.5) - 0.5
        edges_idx = np.unique(edges_idx)
        edges_ref = self.pix_to_coord(edges_idx)

        groups = Table()
        groups[f"{self.name}_min"] = edges_ref[:-1]
        groups[f"{self.name}_max"] = edges_ref[1:]

        groups["idx_min"] = (edges_idx[:-1] + 0.5).astype(int)
        groups["idx_max"] = (edges_idx[1:] - 0.5).astype(int)

        if len(groups) == 0:
            raise ValueError("No overlap between reference and target edges.")

        groups["bin_type"] = "normal   "

        edge_idx_start, edge_ref_start = edges_idx[0], edges_ref[0]
        if edge_idx_start > 0:
            underflow = {
                "bin_type": "underflow",
                "idx_min": 0,
                "idx_max": edge_idx_start,
                f"{self.name}_min": self.pix_to_coord(-0.5),
                f"{self.name}_max": edge_ref_start,
            }
            groups.insert_row(0, vals=underflow)

        edge_idx_end, edge_ref_end = edges_idx[-1], edges_ref[-1]

        if edge_idx_end < (self.nbin - 0.5):
            overflow = {
                "bin_type": "overflow",
                "idx_min": edge_idx_end + 1,
                "idx_max": self.nbin - 1,
                f"{self.name}_min": edge_ref_end,
                f"{self.name}_max": self.pix_to_coord(self.nbin - 0.5),
            }
            groups.add_row(vals=overflow)

        group_idx = Column(np.arange(len(groups)))
        groups.add_column(group_idx, name="group_idx", index=0)
        return groups

    def upsample(self, factor):
        """Upsample map axis by a given factor.

        When up-sampling for each node specified in the axis, the corresponding
        number of sub-nodes are introduced and preserving the initial nodes. For
        node type "edges" this results in nbin * factor new bins. For node type
        "center" this results in (nbin - 1) * factor + 1 new bins.

        Parameters
        ----------
        factor : int
            Upsampling factor.

        Returns
        -------
        axis : `MapAxis`
            Usampled map axis.

        """
        if self.node_type == "edges":
            pix = self.coord_to_pix(self.edges)
            nbin = int(self.nbin * factor) + 1
            pix_new = np.linspace(pix.min(), pix.max(), nbin)
            edges = self.pix_to_coord(pix_new)
            return self.from_edges(edges, name=self.name, interp=self.interp)
        else:
            pix = self.coord_to_pix(self.center)
            nbin = int((self.nbin - 1) * factor) + 1
            pix_new = np.linspace(pix.min(), pix.max(), nbin)
            nodes = self.pix_to_coord(pix_new)
            return self.from_nodes(nodes, name=self.name, interp=self.interp)

    def downsample(self, factor):
        """Downsample map axis by a given factor.

        When down-sampling each n-th (given by the factor) bin is selected from
        the axis while preserving the axis limits. For node type "edges" this
        requires nbin to be dividable by the factor, for node type "center" this
        requires nbin - 1 to be dividable by the factor.

        Parameters
        ----------
        factor : int
            Downsampling factor.


        Returns
        -------
        axis : `MapAxis`
            Downsampled map axis.
        """
        if self.node_type == "edges":
            nbin = self.nbin / factor

            if np.mod(nbin, 1) > 0:
                raise ValueError(
                    f"Number of {self.name} bins is not divisible by {factor}"
                )

            edges = self.edges[::factor]
            return self.from_edges(edges, name=self.name, interp=self.interp)
        else:
            nbin = (self.nbin - 1) / factor

            if np.mod(nbin, 1) > 0:
                raise ValueError(
                    f"Number of {self.name} bins - 1 is not divisible by {factor}"
                )

            nodes = self.center[::factor]
            return self.from_nodes(nodes, name=self.name, interp=self.interp)

    def to_header(self, header, format="ogip"):
        """Create FITS header

        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            Header to extend.
        format : {"ogip"}
            Format specification

        Returns
        -------
        header : `~astropy.io.fits.Header`
            Header to extend.
        """

        if format == "ogip":
            header["EXTNAME"] = "EBOUNDS", "Name of this binary table extension"
            header["TELESCOP"] = "DUMMY", "Mission/satellite name"
            header["INSTRUME"] = "DUMMY", "Instrument/detector"
            header["FILTER"] = "None", "Filter information"
            header["CHANTYPE"] = "PHA", "Type of channels (PHA, PI etc)"
            header["DETCHANS"] = self.nbin, "Total number of detector PHA channels"
            header["HDUCLASS"] = "OGIP", "Organisation devising file format"
            header["HDUCLAS1"] = "RESPONSE", "File relates to response of instrument"
            header["HDUCLAS2"] = "EBOUNDS", "This is an EBOUNDS extension"
            header["HDUVERS"] = "1.2.0", "Version of file format"
        else:
            raise ValueError(f"Unknown format {format}")

        return header

    def to_table(self, format="ogip"):
        """Convert `~astropy.units.Quantity` to OGIP ``EBOUNDS`` extension.

        See https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html#tth_sEc3.2

        The 'ogip-sherpa' format is equivalent to 'ogip' but uses keV energy units.

        Parameters
        ----------
        format : {"ogip", "ogip-sherpa", "gadf-dl3"}
            Format specification

        Returns
        -------
        table : `~astropy.table.Table`
            Table HDU
        """
        table = Table()

        if format in ["ogip", "ogip-sherpa"]:
            if "energy" not in self.name:
                raise ValueError("Only energy axes can be converted to HDU")

            edges = self.edges

            if format == "ogip-sherpa":
                edges = edges.to("keV")

            table["CHANNEL"] = np.arange(self.nbin, dtype=np.int16)
            table["E_MIN"] = edges[:-1]
            table["E_MAX"] = edges[1:]
        elif format == "gadf-dl3":
            from gammapy.irf.io import IRF_DL3_AXES_SPECIFICATION

            if self.name == "energy":
                column_prefix = "ENERG"
            else:
                for column_prefix, spec in IRF_DL3_AXES_SPECIFICATION.items():
                    if spec["name"] == self.name:
                        break

            if self.node_type == "edges":
                edges_hi, edges_lo = self.edges[:-1], self.edges[1:]
            else:
                edges_hi, edges_lo = self.center, self.center

            table[f"{column_prefix}_LO"] = edges_hi[np.newaxis]
            table[f"{column_prefix}_HI"] = edges_lo[np.newaxis]
        else:
            raise ValueError(f"{format} is not a valid format")

        return table

    def to_table_hdu(self, format="ogip"):
        """Convert `~astropy.units.Quantity` to OGIP ``EBOUNDS`` extension.

        See https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html#tth_sEc3.2

        The 'ogip-sherpa' format is equivalent to 'ogip' but uses keV energy units.

        Parameters
        ----------
        format : {"ogip", "ogip-sherpa"}
            Format specification

        Returns
        -------
        hdu : `~astropy.io.fits.BinTableHDU`
            Table HDU
        """
        table = self.to_table(format=format)
        hdu = fits.BinTableHDU(table)
        if format in ["ogip", "ogip-sherpa"]:
            hdu.header = self.to_header(hdu.header, format=format)
        return hdu

    @classmethod
    def from_table(cls, table, format="ogip", idx=0, column_prefix=""):
        """Instanciate MapAxis from table HDU

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table
        format : {"ogip", "ogip-arf", "fgst-ccube", "fgst-template", "gadf", "gadf-dl3"}
            Format specification
        idx : int
            Column index of the axis.
        column_prefix : str
            Column name prefix of the axis, used for

        Returns
        -------
        axis : `MapAxis`
            Map Axis
        """
        if format in ["ogip", "fgst-ccube"]:
            energy_min = table["E_MIN"].quantity
            energy_max = table["E_MAX"].quantity
            energy_edges = (
                np.append(energy_min.value, energy_max.value[-1]) * energy_min.unit
            )
            axis = cls.from_edges(energy_edges, name="energy", interp="log")

        elif format == "ogip-arf":
            energy_min = table["ENERG_LO"].quantity
            energy_max = table["ENERG_HI"].quantity
            energy_edges = (
                np.append(energy_min.value, energy_max.value[-1]) * energy_min.unit
            )
            axis = cls.from_edges(energy_edges, name="energy_true", interp="log")

        elif format in ["fgst-template", "fgst-bexpcube"]:
            allowed_names = ["Energy", "ENERGY", "energy"]
            for colname in table.colnames:
                if colname in allowed_names:
                    tag = colname
                    break

            nodes = table[tag].data
            axis = cls.from_nodes(
                nodes=nodes, name="energy_true", unit="MeV", interp="log"
            )

        elif format == "gadf":
            axcols = table.meta.get("AXCOLS{}".format(idx + 1))
            colnames = axcols.split(",")
            node_type = "edges" if len(colnames) == 2 else "center"

            # TODO: check why this extra case is needed
            if colnames[0] == "E_MIN":
                name = "energy"
            else:
                name = colnames[0].replace("_MIN", "").lower()
                # this is need for backward compatibility
                if name == "theta":
                    name = "rad"

            interp = table.meta.get("INTERP{}".format(idx + 1), "lin")

            if node_type == "center":
                nodes = np.unique(table[colnames[0]].quantity)
            else:
                edges_min = np.unique(table[colnames[0]].quantity)
                edges_max = np.unique(table[colnames[1]].quantity)
                nodes = edges_from_lo_hi(edges_min, edges_max)

            axis = MapAxis(nodes=nodes, node_type=node_type, interp=interp, name=name)

        elif format == "gadf-dl3":
            from gammapy.irf.io import IRF_DL3_AXES_SPECIFICATION

            spec = IRF_DL3_AXES_SPECIFICATION[column_prefix]
            name, interp = spec["name"], spec["interp"]

            # background models are stored in reconstructed energy
            extname = table.meta.get("EXTNAME")
            if extname in ["BACKGROUND", "BKG"] and column_prefix == "ENERG":
                name = "energy"

            edges_lo = table[f"{column_prefix}_LO"].quantity[0]
            edges_hi = table[f"{column_prefix}_HI"].quantity[0]

            if np.allclose(edges_hi, edges_lo):
                axis = MapAxis.from_nodes(edges_hi, interp=interp, name=name)
            else:
                edges = edges_from_lo_hi(edges_lo, edges_hi)
                axis = MapAxis.from_edges(edges, interp=interp, name=name)
        else:
            raise ValueError(f"Format '{format}' not supported")

        return axis

    @classmethod
    def from_table_hdu(cls, hdu, format="ogip", idx=0):
        """Instanciate MapAxis from table HDU

        Parameters
        ----------
        hdu : `~astropy.io.fits.BinTableHDU`
            Table HDU
        format : {"ogip", "ogip-arf", "fgst-ccube", "fgst-template"}
            Format specification
        idx : int
            Column index of the axis.

        Returns
        -------
        axis : `MapAxis`
            Map Axis
        """
        table = Table.read(hdu)
        return cls.from_table(table, format=format, idx=idx)


class MapCoord:
    """Represents a sequence of n-dimensional map coordinates.

    Contains coordinates for 2 spatial dimensions and an arbitrary
    number of additional non-spatial dimensions.

    For further information see :ref:`mapcoord`.

    Parameters
    ----------
    data : `dict` of `~numpy.ndarray`
        Dictionary of coordinate arrays.
    frame : {"icrs", "galactic", None}
        Spatial coordinate system.  If None then the coordinate system
        will be set to the native coordinate system of the geometry.
    match_by_name : bool
        Match coordinates to axes by name?
        If false coordinates will be matched by index.
    """

    def __init__(self, data, frame=None, match_by_name=True):

        if "lon" not in data or "lat" not in data:
            raise ValueError("data dictionary must contain axes named 'lon' and 'lat'.")

        data = {k: np.atleast_1d(np.asanyarray(v)) for k, v in data.items()}
        vals = np.broadcast_arrays(*data.values(), subok=True)
        self._data = dict(zip(data.keys(), vals))
        self._frame = frame
        self._match_by_name = match_by_name

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._data[key]
        else:
            return list(self._data.values())[key]

    def __iter__(self):
        return iter(self._data.values())

    @property
    def ndim(self):
        """Number of dimensions."""
        return len(self._data)

    @property
    def shape(self):
        """Coordinate array shape."""
        return self[0].shape

    @property
    def size(self):
        return self[0].size

    @property
    def lon(self):
        """Longitude coordinate in degrees."""
        return self._data["lon"]

    @property
    def lat(self):
        """Latitude coordinate in degrees."""
        return self._data["lat"]

    @property
    def theta(self):
        """Theta co-latitude angle in radians."""
        theta = u.Quantity(self.lat, unit="deg", copy=False).to_value("rad")
        return np.pi / 2.0 - theta

    @property
    def phi(self):
        """Phi longitude angle in radians."""
        phi = u.Quantity(self.lon, unit="deg", copy=False).to_value("rad")
        return phi

    @property
    def frame(self):
        """Coordinate system (str)."""
        return self._frame

    @property
    def match_by_name(self):
        """Boolean flag: axis lookup by name (True) or index (False)."""
        return self._match_by_name

    @property
    def skycoord(self):
        return SkyCoord(self.lon, self.lat, unit="deg", frame=self.frame)

    @classmethod
    def _from_lonlat(cls, coords, frame=None, axis_names=None):
        """Create a `~MapCoord` from a tuple of coordinate vectors.

        The first two elements of the tuple should be longitude and latitude in degrees.

        Parameters
        ----------
        coords : tuple
            Tuple of `~numpy.ndarray`.

        Returns
        -------
        coord : `~MapCoord`
            A coordinates object.
        """
        if axis_names is None:
            axis_names = [f"axis{idx}" for idx in range(len(coords) - 2)]

        if isinstance(coords, (list, tuple)):
            coords_dict = {"lon": coords[0], "lat": coords[1]}
            for name, c in zip(axis_names, coords[2:]):
                coords_dict[name] = c
        else:
            raise ValueError("Unrecognized input type.")

        return cls(coords_dict, frame=frame, match_by_name=False)

    @classmethod
    def _from_tuple(cls, coords, frame=None, axis_names=None):
        """Create from tuple of coordinate vectors."""
        if isinstance(coords[0], (list, np.ndarray)) or np.isscalar(coords[0]):
            return cls._from_lonlat(coords, frame=frame, axis_names=axis_names)
        elif isinstance(coords[0], SkyCoord):
            lon, lat, frame = skycoord_to_lonlat(coords[0], frame=frame)
            coords = (lon, lat) + coords[1:]
            return cls._from_lonlat(coords, frame=frame, axis_names=axis_names)
        else:
            raise TypeError(f"Type not supported: {type(coords)!r}")

    @classmethod
    def _from_dict(cls, coords, frame=None):
        """Create from a dictionary of coordinate vectors."""
        if "lon" in coords and "lat" in coords:
            return cls(coords, frame=frame)
        elif "skycoord" in coords:
            lon, lat, frame = skycoord_to_lonlat(coords["skycoord"], frame=frame)
            coords_dict = {"lon": lon, "lat": lat}
            for k, v in coords.items():
                if k == "skycoord":
                    continue
                coords_dict[k] = v
            return cls(coords_dict, frame=frame)
        else:
            raise ValueError("coords dict must contain 'lon'/'lat' or 'skycoord'.")

    @classmethod
    def create(cls, data, frame=None, axis_names=None):
        """Create a new `~MapCoord` object.

        This method can be used to create either unnamed (with tuple input)
        or named (via dict input) axes.

        Parameters
        ----------
        data : tuple, dict, `MapCoord` or `~astropy.coordinates.SkyCoord`
            Object containing coordinate arrays.
        frame : {"icrs", "galactic", None}, optional
            Set the coordinate system for longitude and latitude. If
            None longitude and latitude will be assumed to be in
            the coordinate system native to a given map geometry.
        axis_names : list of str
            Axis names use if a tuple is provided

        Examples
        --------
        >>> from astropy.coordinates import SkyCoord
        >>> from gammapy.maps import MapCoord

        >>> lon, lat = [1, 2], [2, 3]
        >>> skycoord = SkyCoord(lon, lat, unit='deg')
        >>> energy = [1000]
        >>> c = MapCoord.create((lon,lat))
        >>> c = MapCoord.create((skycoord,))
        >>> c = MapCoord.create((lon,lat,energy))
        >>> c = MapCoord.create(dict(lon=lon,lat=lat))
        >>> c = MapCoord.create(dict(lon=lon,lat=lat,energy=energy))
        >>> c = MapCoord.create(dict(skycoord=skycoord,energy=energy))
        """
        if isinstance(data, cls):
            if data.frame is None or frame == data.frame:
                return data
            else:
                return data.to_frame(frame)
        elif isinstance(data, dict):
            return cls._from_dict(data, frame=frame)
        elif isinstance(data, (list, tuple)):
            return cls._from_tuple(data, frame=frame, axis_names=axis_names)
        elif isinstance(data, SkyCoord):
            return cls._from_tuple((data,), frame=frame, axis_names=axis_names)
        else:
            raise TypeError(f"Unsupported input type: {type(data)!r}")

    def to_frame(self, frame):
        """Convert to a different coordinate frame.

        Parameters
        ----------
        frame : {"icrs", "galactic"}
            Coordinate system, either Galactic ("galactic") or Equatorial ("icrs").

        Returns
        -------
        coords : `~MapCoord`
            A coordinates object.
        """
        if frame == self.frame:
            return copy.deepcopy(self)
        else:
            lon, lat, frame = skycoord_to_lonlat(self.skycoord, frame=frame)
            data = copy.deepcopy(self._data)
            if isinstance(self.lon, u.Quantity):
                lon = u.Quantity(lon, unit="deg", copy=False)

            if isinstance(self.lon, u.Quantity):
                lat = u.Quantity(lat, unit="deg", copy=False)

            data["lon"] = lon
            data["lat"] = lat
            return self.__class__(data, frame, self._match_by_name)

    def apply_mask(self, mask):
        """Return a masked copy of this coordinate object.

        Parameters
        ----------
        mask : `~numpy.ndarray`
            Boolean mask.

        Returns
        -------
        coords : `~MapCoord`
            A coordinates object.
        """
        data = {k: v[mask] for k, v in self._data.items()}
        return self.__class__(data, self.frame, self._match_by_name)

    def copy(self):
        """Copy `MapCoord` object."""
        return copy.deepcopy(self)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}\n\n"
            f"\taxes     : {list(self._data.keys())}\n"
            f"\tshape    : {self.shape[::-1]}\n"
            f"\tndim     : {self.ndim}\n"
            f"\tframe : {self.frame}\n"
        )


class Geom(abc.ABC):
    """Map geometry base class.

    See also: `~gammapy.maps.WcsGeom` and `~gammapy.maps.HpxGeom`
    """

    @property
    @abc.abstractmethod
    def data_shape(self):
        """Shape of the Numpy data array matching this geometry."""
        pass

    @property
    @abc.abstractmethod
    def is_allsky(self):
        pass

    @property
    @abc.abstractmethod
    def center_coord(self):
        pass

    @property
    @abc.abstractmethod
    def center_pix(self):
        pass

    @property
    @abc.abstractmethod
    def center_skydir(self):
        pass

    @classmethod
    def from_hdulist(cls, hdulist, hdu=None, hdu_bands=None):
        """Load a geometry object from a FITS HDUList.

        Parameters
        ----------
        hdulist :  `~astropy.io.fits.HDUList`
            HDU list containing HDUs for map data and bands.
        hdu : str
            Name or index of the HDU with the map data.
        hdu_bands : str
            Name or index of the HDU with the BANDS table.  If not
            defined this will be inferred from the FITS header of the
            map HDU.

        Returns
        -------
        geom : `~Geom`
            Geometry object.
        """
        if hdu is None:
            hdu = find_hdu(hdulist)
        else:
            hdu = hdulist[hdu]

        if hdu_bands is None:
            hdu_bands = find_bands_hdu(hdulist, hdu)

        if hdu_bands is not None:
            hdu_bands = hdulist[hdu_bands]

        return cls.from_header(hdu.header, hdu_bands)

    def to_bands_hdu(self, hdu=None, hdu_skymap=None, format=None):
        table_hdu = self.axes.to_table_hdu(format=format, prefix=hdu_skymap)
        cols = table_hdu.columns.columns
        cols.extend(self._make_bands_cols())
        return fits.BinTableHDU.from_columns(
            cols, header=table_hdu.header, name=table_hdu.name
        )

    @abc.abstractmethod
    def _make_bands_cols(self):
        pass

    @abc.abstractmethod
    def get_idx(self, idx=None, local=False, flat=False):
        """Get tuple of pixel indices for this geometry.

        Returns all pixels in the geometry by default. Pixel indices
        for a single image plane can be accessed by setting ``idx``
        to the index tuple of a plane.

        Parameters
        ----------
        idx : tuple, optional
            A tuple of indices with one index for each non-spatial
            dimension.  If defined only pixels for the image plane with
            this index will be returned.  If none then all pixels
            will be returned.
        local : bool
            Flag to return local or global pixel indices.  Local
            indices run from 0 to the number of pixels in a given
            image plane.
        flat : bool, optional
            Return a flattened array containing only indices for
            pixels contained in the geometry.

        Returns
        -------
        idx : tuple
            Tuple of pixel index vectors with one vector for each
            dimension.
        """
        pass

    @abc.abstractmethod
    def get_coord(self, idx=None, flat=False):
        """Get the coordinate array for this geometry.

        Returns a coordinate array with the same shape as the data
        array.  Pixels outside the geometry are set to NaN.
        Coordinates for a single image plane can be accessed by
        setting ``idx`` to the index tuple of a plane.

        Parameters
        ----------
        idx : tuple, optional
            A tuple of indices with one index for each non-spatial
            dimension.  If defined only coordinates for the image
            plane with this index will be returned.  If none then
            coordinates for all pixels will be returned.
        flat : bool, optional
            Return a flattened array containing only coordinates for
            pixels contained in the geometry.

        Returns
        -------
        coords : tuple
            Tuple of coordinate vectors with one vector for each
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
            either be a tuple of numpy arrays or a MapCoord object.
            If passed as a tuple then the ordering should be
            (longitude, latitude, c_0, ..., c_N) where c_i is the
            coordinate vector for axis i.

        Returns
        -------
        pix : tuple
            Tuple of pixel coordinates in image and band dimensions.
        """
        pass

    def coord_to_idx(self, coords, clip=False):
        """Convert map coordinates to pixel indices.

        Parameters
        ----------
        coords : tuple or `~MapCoord`
            Coordinate values in each dimension of the map.  This can
            either be a tuple of numpy arrays or a MapCoord object.
            If passed as a tuple then the ordering should be
            (longitude, latitude, c_0, ..., c_N) where c_i is the
            coordinate vector for axis i.
        clip : bool
            Choose whether to clip indices to the valid range of the
            geometry.  If false then indices for coordinates outside
            the geometry range will be set -1.

        Returns
        -------
        pix : tuple
            Tuple of pixel indices in image and band dimensions.
            Elements set to -1 correspond to coordinates outside the
            map.
        """
        pix = self.coord_to_pix(coords)
        return self.pix_to_idx(pix, clip=clip)

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
    def pix_to_idx(self, pix, clip=False):
        """Convert pixel coordinates to pixel indices.

        Returns -1 for pixel coordinates that lie outside of the map.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel coordinates.
        clip : bool
            Choose whether to clip indices to the valid range of the
            geometry.  If false then indices for coordinates outside
            the geometry range will be set -1.

        Returns
        -------
        idx : tuple
            Tuple of pixel indices.
        """
        pass

    @abc.abstractmethod
    def contains(self, coords):
        """Check if a given map coordinate is contained in the geometry.

        Parameters
        ----------
        coords : tuple or `~gammapy.maps.MapCoord`
            Tuple of map coordinates.

        Returns
        -------
        containment : `~numpy.ndarray`
            Bool array.
        """
        pass

    def contains_pix(self, pix):
        """Check if a given pixel coordinate is contained in the geometry.

        Parameters
        ----------
        pix : tuple
            Tuple of pixel coordinates.

        Returns
        -------
        containment : `~numpy.ndarray`
            Bool array.
        """
        idx = self.pix_to_idx(pix)
        return np.all(np.stack([t != INVALID_INDEX.int for t in idx]), axis=0)

    def slice_by_idx(self, slices):
        """Create a new geometry by slicing the non-spatial axes.

        Parameters
        ----------
        slices : dict
            Dict of axes names and integers or `slice` object pairs. Contains one
            element for each non-spatial dimension. For integer indexing the
            corresponding axes is dropped from the map. Axes not specified in the
            dict are kept unchanged.

        Returns
        -------
        geom : `~Geom`
            Sliced geometry.
        """
        axes = self.axes.slice_by_idx(slices)
        return self._init_copy(axes=axes)

    @property
    def as_energy_true(self):
        """If the geom contains an energy axis rename it to energy true"""
        energy_axis = self.axes["energy"].copy(name="energy_true")
        return self.to_image().to_cube([energy_axis])

    @abc.abstractmethod
    def to_image(self):
        """Create 2D image geometry (drop non-spatial dimensions).

        Returns
        -------
        geom : `~Geom`
            Image geometry.
        """
        pass

    @abc.abstractmethod
    def to_cube(self, axes):
        """Append non-spatial axes to create a higher-dimensional geometry.

        This will result in a new geometry with
        N+M dimensions where N is the number of current dimensions and
        M is the number of axes in the list.

        Parameters
        ----------
        axes : list
            Axes that will be appended to this geometry.

        Returns
        -------
        geom : `~Geom`
            Map geometry.
        """
        pass

    def squash(self, axis_name):
        """Squash geom axis.

        Parameters
        ----------
        axis_name : str
            Axis to squash.

        Returns
        -------
        geom : `Geom`
            Geom with squashed axis.
        """
        axes = self.axes.squash(axis_name=axis_name)
        return self.to_image().to_cube(axes=axes)

    def drop(self, axis_name):
        """Drop an axis from the geom.

        Parameters
        ----------
        axis_name : str
            Name of the axis to remove.

        Returns
            -------
        geom : `Geom`
            New geom with the axis removed.
        """
        axes = self.axes.drop(axis_name=axis_name)
        return self.to_image().to_cube(axes=axes)

    @abc.abstractmethod
    def pad(self, pad_width):
        """
        Pad the geometry at the edges.

        Parameters
        ----------
        pad_width : {sequence, array_like, int}
            Number of values padded to the edges of each axis.

        Returns
        -------
        geom : `~Geom`
            Padded geometry.
        """
        pass

    @abc.abstractmethod
    def crop(self, crop_width):
        """
        Crop the geometry at the edges.

        Parameters
        ----------
        crop_width : {sequence, array_like, int}
            Number of values cropped from the edges of each axis.

        Returns
        -------
        geom : `~Geom`
            Cropped geometry.
        """
        pass

    @abc.abstractmethod
    def downsample(self, factor, axis_name):
        """Downsample the spatial dimension of the geometry by a given factor.

        Parameters
        ----------
        factor : int
            Downsampling factor.
        axis_name : str
            Axis to downsample.

        Returns
        -------
        geom : `~Geom`
            Downsampled geometry.

        """
        pass

    @abc.abstractmethod
    def upsample(self, factor, axis_name):
        """Upsample the spatial dimension of the geometry by a given factor.

        Parameters
        ----------
        factor : int
            Upsampling factor.
        axis_name : str
            Axis to upsample.

        Returns
        -------
        geom : `~Geom`
            Upsampled geometry.

        """
        pass

    def resample_axis(self, axis):
        """Resample geom to a new axis binning.

        This method groups the existing bins into a new binning.

        Parameters
        ----------
        axis : `MapAxis`
            New map axis.

        Returns
        -------
        map : `Geom`
            Geom with resampled axis.
        """
        axes = self.axes.resample(axis=axis)
        return self._init_copy(axes=axes)

    @abc.abstractmethod
    def solid_angle(self):
        """Solid angle (`~astropy.units.Quantity` in ``sr``)."""
        pass

    @property
    def is_image(self):
        """Whether the geom is an image without extra dimensions."""
        if self.axes is None:
            return True
        return len(self.axes) == 0

    @property
    def is_flat(self):
        """Whether the geom non spatial axes have length 1, i.e. if the geom is equivalent to an image."""
        if self.is_image:
            return True
        else:
            valid = True
            for axis in self.axes:
                valid = valid and (axis.nbin == 1)
            return valid

    def _init_copy(self, **kwargs):
        """Init map geom instance by copying missing init arguments from self.
        """
        argnames = inspect.getfullargspec(self.__init__).args
        argnames.remove("self")

        for arg in argnames:
            value = getattr(self, "_" + arg)
            kwargs.setdefault(arg, copy.deepcopy(value))

        return self.__class__(**kwargs)

    def copy(self, **kwargs):
        """Copy and overwrite given attributes.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to overwrite in the map geometry constructor.

        Returns
        -------
        copy : `Geom`
            Copied map geometry.
        """
        return self._init_copy(**kwargs)

    def energy_mask(self, energy_min=None, energy_max=None, round_to_edge=False):
        """Create a mask for a given energy range.

        The energy bin must be fully contained to be included in the mask.

        Parameters
        ----------
        energy_min, energy_max : `~astropy.units.Quantity`
            Energy range

        Returns
        -------
        mask : `~numpy.ndarray`
            Energy mask
        """
        # get energy axes and values
        energy_axis = self.axes["energy"]

        if round_to_edge:
            energy_min, energy_max = energy_axis.round([energy_min, energy_max])

        # TODO: make this more general
        shape = (-1, 1) if self.is_hpx else (-1, 1, 1)
        energy_edges = energy_axis.edges.reshape(shape)

        # set default values
        energy_min = energy_min if energy_min is not None else energy_edges[0]
        energy_max = energy_max if energy_max is not None else energy_edges[-1]

        mask = (energy_edges[:-1] >= energy_min) & (energy_edges[1:] <= energy_max)
        return np.broadcast_to(mask, shape=self.data_shape)

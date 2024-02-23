# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import html
import inspect
import logging
from collections.abc import Sequence
import numpy as np
import scipy
import astropy.units as u
from astropy.io import fits
from astropy.table import Column, Table, hstack
from astropy.time import Time
from astropy.utils import lazyproperty
import matplotlib.pyplot as plt
from gammapy.utils.interpolation import interpolation_scale
from gammapy.utils.time import time_ref_from_dict, time_ref_to_dict
from .utils import INVALID_INDEX, INVALID_VALUE, edges_from_lo_hi

__all__ = ["MapAxes", "MapAxis", "TimeMapAxis", "LabelMapAxis"]

log = logging.getLogger(__name__)


def flat_if_equal(array):
    if array.ndim == 2 and np.all(array == array[0]):
        return array[0]
    else:
        return array


class AxisCoordInterpolator:
    """Axis coordinate interpolator."""

    def __init__(self, edges, interp="lin"):
        self.scale = interpolation_scale(interp)
        self.x = self.scale(edges)
        self.y = np.arange(len(edges), dtype=float)
        self.fill_value = "extrapolate"

        if len(edges) == 1:
            self.kind = 0
        else:
            self.kind = 1

    def coord_to_pix(self, coord):
        """Transform coordinate to pixel."""
        interp_fn = scipy.interpolate.interp1d(
            x=self.x, y=self.y, kind=self.kind, fill_value=self.fill_value
        )
        return interp_fn(self.scale(coord))

    def pix_to_coord(self, pix):
        """Transform pixel to coordinate."""
        interp_fn = scipy.interpolate.interp1d(
            x=self.y, y=self.x, kind=self.kind, fill_value=self.fill_value
        )
        return self.scale.inverse(interp_fn(pix))


PLOT_AXIS_LABEL = {
    "energy": "Energy",
    "energy_true": "True Energy",
    "offset": "FoV Offset",
    "rad": "Source Offset",
    "migra": "Energy / True Energy",
    "fov_lon": "FoV Lon.",
    "fov_lat": "FoV Lat.",
    "time": "Time",
}

DEFAULT_LABEL_TEMPLATE = "{quantity} [{unit}]"
UNIT_STRING_FORMAT = "latex_inline"


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
    interp : {'lin', 'log', 'sqrt'}
        Interpolation method used to transform between axis and pixel
        coordinates. Default is 'lin'.
    name : str, optional
        Axis name. Default is "".
    node_type : str, optional
        Flag indicating whether coordinate nodes correspond to pixel
        edges (node_type = 'edges') or pixel centers (node_type =
        'center').  'center' should be used where the map values are
        defined at a specific coordinate (e.g. differential
        quantities). 'edges' should be used where map values are
        defined by an integral over coordinate intervals (e.g. a
        counts histogram). Default is "edges".
    unit : str, optional
        String specifying the data units. Default is "".
    """

    # TODO: Cache an interpolation object?
    def __init__(self, nodes, interp="lin", name="", node_type="edges", unit=""):
        if not isinstance(name, str):
            raise TypeError(f"Name must be a string, got: {type(name)!r}")

        if len(nodes) != len(np.unique(nodes)):
            raise ValueError("MapAxis: node values must be unique")

        if ~(np.all(nodes == np.sort(nodes)) or np.all(nodes[::-1] == np.sort(nodes))):
            raise ValueError("MapAxis: node values must be sorted")

        if isinstance(nodes, u.Quantity):
            unit = nodes.unit if nodes.unit is not None else ""
            nodes = nodes.value
        else:
            nodes = np.array(nodes)

        self._name = name
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
        self._use_center_as_plot_labels = None

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    def assert_name(self, required_name):
        """Assert axis name if a specific one is required.

        Parameters
        ----------
        required_name : str
            Required name.
        """
        if self.name != required_name:
            raise ValueError(
                "Unexpected axis name,"
                f' expected "{required_name}", got: "{self.name}"'
            )

    def is_aligned(self, other, atol=2e-2):
        """Check if the other map axis is aligned.

        Two axes are aligned if their center coordinate values map to integers
        on the other axes as well and if the interpolation modes are equivalent.

        Parameters
        ----------
        other : `MapAxis`
            Other map axis.
        atol : float, optional
            Absolute numerical tolerance for the comparison measured in bins. Default is 2e-2.

        Returns
        -------
        aligned : bool
            Whether the axes are aligned.
        """
        pix = self.coord_to_pix(other.center)
        pix_other = other.coord_to_pix(self.center)
        pix_all = np.append(pix, pix_other)
        aligned = np.allclose(np.round(pix_all) - pix_all, 0, atol=atol)
        return aligned and self.interp == other.interp

    def is_allclose(self, other, **kwargs):
        """Check if the other map axis is all close.

        Parameters
        ----------
        other : `MapAxis`
            Other map axis.
        **kwargs : dict, optional
            Keyword arguments passed to `~numpy.allclose`.

        Returns
        -------
        is_allclose : bool
            Whether the other axis is allclose.
        """
        if not isinstance(other, self.__class__):
            return TypeError(f"Cannot compare {type(self)} and {type(other)}")

        if self.edges.shape != other.edges.shape:
            return False
        if not self.unit.is_equivalent(other.unit):
            return False
        return (
            np.allclose(self.edges, other.edges, **kwargs)
            and self._node_type == other._node_type
            and self._interp == other._interp
            and self.name.upper() == other.name.upper()
        )

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.is_allclose(other, rtol=1e-6, atol=1e-6)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return id(self)

    @lazyproperty
    def _transform(self):
        """Interpolate coordinates to pixel."""
        return AxisCoordInterpolator(edges=self._nodes, interp=self.interp)

    @property
    def is_energy_axis(self):
        """Whether this is an energy axis."""
        return self.name in ["energy", "energy_true"]

    @property
    def interp(self):
        """Interpolation scale of the axis."""
        return self._interp

    @property
    def name(self):
        """Name of the axis."""
        return self._name

    @lazyproperty
    def edges(self):
        """Return an array of bin edges."""
        pix = np.arange(self.nbin + 1, dtype=float) - 0.5
        return u.Quantity(self.pix_to_coord(pix), self._unit, copy=False)

    @property
    def edges_min(self):
        """Return an array of bin edges maximum values."""
        return self.edges[:-1]

    @property
    def edges_max(self):
        """Return an array of bin edges minimum values."""
        return self.edges[1:]

    @property
    def bounds(self):
        """Bounds of the axis as a `~astropy.units.Quantity`."""
        idx = [0, -1]
        if self.node_type == "edges":
            return self.edges[idx]
        else:
            return self.center[idx]

    @property
    def as_plot_xerr(self):
        """Return a tuple of x-error to be passed to `~matplotlib.pyplot.errorbar`."""
        return (
            self.center - self.edges_min,
            self.edges_max - self.center,
        )

    @property
    def use_center_as_plot_labels(self):
        """Use center as plot labels."""
        if self._use_center_as_plot_labels is not None:
            return self._use_center_as_plot_labels

        return self.node_type == "center"

    @use_center_as_plot_labels.setter
    def use_center_as_plot_labels(self, value):
        """Use center as plot labels."""
        self._use_center_as_plot_labels = bool(value)

    @property
    def as_plot_labels(self):
        """Return a list of axis plot labels."""
        if self.use_center_as_plot_labels:
            labels = [f"{val:.2e}" for val in self.center]
        else:
            labels = [
                f"{val_min:.2e} - {val_max:.2e}"
                for val_min, val_max in self.iter_by_edges
            ]
        return labels

    @property
    def as_plot_edges(self):
        """Plot edges."""
        return self.edges

    @property
    def as_plot_center(self):
        """Plot center."""
        return self.center

    @property
    def as_plot_scale(self):
        """Plot axis scale."""
        mpl_scale = {"lin": "linear", "sqrt": "linear", "log": "log"}

        return mpl_scale[self.interp]

    def to_node_type(self, node_type):
        """Return a copy of the `MapAxis` instance with a node type set to node_type.

        Parameters
        ----------
        node_type : str
            The target node type. It can be either 'center' or 'edges'.

        Returns
        -------
        axis : `~gammapy.maps.MapAxis`
            The new MapAxis.
        """
        if node_type == self.node_type:
            return self
        else:
            if node_type == "center":
                nodes = self.center
            else:
                nodes = self.edges
            return self.__class__(
                nodes=nodes,
                interp=self.interp,
                name=self.name,
                node_type=node_type,
                unit=self.unit,
            )

    def rename(self, new_name):
        """Rename the axis. Return a copy of the `MapAxis` instance with name set to new_name.

        Parameters
        ----------
        new_name : str
            The new name for the axis.

        Returns
        -------
        axis : `~gammapy.maps.MapAxis`
            The new MapAxis.
        """
        return self.copy(name=new_name)

    def format_plot_xaxis(self, ax):
        """Format the x-axis.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axis`
            Plot axis to format.

        Returns
        -------
        ax : `~matplotlib.pyplot.Axis`
            Formatted plot axis.
        """
        ax.set_xscale(self.as_plot_scale)

        xlabel = DEFAULT_LABEL_TEMPLATE.format(
            quantity=PLOT_AXIS_LABEL.get(self.name, self.name.capitalize()),
            unit=ax.xaxis.units.to_string(UNIT_STRING_FORMAT),
        )
        ax.set_xlabel(xlabel)
        xmin, xmax = self.bounds
        if not xmin == xmax:
            ax.set_xlim(self.bounds)
        return ax

    def format_plot_yaxis(self, ax):
        """Format plot y-axis.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axis`
            Plot axis to format.

        Returns
        -------
        ax : `~matplotlib.pyplot.Axis`
            Formatted plot axis.
        """
        ax.set_yscale(self.as_plot_scale)

        ylabel = DEFAULT_LABEL_TEMPLATE.format(
            quantity=PLOT_AXIS_LABEL.get(self.name, self.name.capitalize()),
            unit=ax.yaxis.units.to_string(UNIT_STRING_FORMAT),
        )
        ax.set_ylabel(ylabel)
        ax.set_ylim(self.bounds)
        return ax

    @property
    def iter_by_edges(self):
        """Iterate by intervals defined by the edges."""
        for value_min, value_max in zip(self.edges[:-1], self.edges[1:]):
            yield (value_min, value_max)

    @lazyproperty
    def center(self):
        """Return an array of bin centers."""
        pix = np.arange(self.nbin, dtype=float)
        return u.Quantity(self.pix_to_coord(pix), self._unit, copy=False)

    @lazyproperty
    def bin_width(self):
        """Array of bin widths."""
        return np.diff(self.edges)

    @property
    def nbin(self):
        """Return the number of bins."""
        return self._nbin

    @property
    def nbin_per_decade(self):
        """Return the number of bins per decade."""
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
        """Return node type, either 'center' or 'edges'."""
        return self._node_type

    @property
    def unit(self):
        """Return the coordinate axis unit."""
        return self._unit

    @classmethod
    def from_bounds(cls, lo_bnd, hi_bnd, nbin, **kwargs):
        """Generate an axis object from a lower/upper bound and number of bins.

        If node_type = 'edges' then bounds correspond to the
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
        ***kwargs : dict, optional
            Keyword arguments passed to `MapAxis`.
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
            nodes = np.geomspace(lo_bnd, hi_bnd, nnode)
        elif interp == "sqrt":
            nodes = np.linspace(lo_bnd**0.5, hi_bnd**0.5, nnode) ** 2.0
        else:
            raise ValueError(f"Invalid interp: {interp}")

        return cls(nodes, **kwargs)

    @classmethod
    def from_energy_edges(cls, energy_edges, unit=None, name=None, interp="log"):
        """Make an energy axis from adjacent edges.

        Parameters
        ----------
        energy_edges : `~astropy.units.Quantity` or float
            Energy edges.
        unit : `~astropy.units.Unit`, optional
            Energy unit. Default is None.
        name : str, optional
            Name of the energy axis, either 'energy' or 'energy_true'. Default is None.
        interp: str, optional
            interpolation mode. Default is 'log'.

        Returns
        -------
        axis : `MapAxis`
            Axis with name "energy" and interp "log".
        """
        energy_edges = u.Quantity(energy_edges, unit)

        if not energy_edges.unit.is_equivalent("TeV"):
            raise ValueError(
                f"Please provide a valid energy unit, got {energy_edges.unit} instead."
            )

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
        strict_bounds=True,
    ):
        """Make an energy axis from energy bounds. The interpolation is always 'log'.

        Used frequently also to make energy grids, by making
        the axis, and then using ``axis.center`` or ``axis.edges``.

        Parameters
        ----------
        energy_min, energy_max : `~astropy.units.Quantity`, float
            Energy range.
        nbin : int
            Number of bins.
        unit : `~astropy.units.Unit`, optional
            Energy unit. Default is None.
        per_decade : bool, optional
            Whether `nbin` is given per decade. Default is False.
        name : str, optional
            Name of the energy axis, either 'energy' or 'energy_true'. Default is None.
        node_type : str, optional
            Node type, either 'edges' or 'center'. Default is 'edges'.
        strict_bounds : bool, optional
            Whether to strictly end the binning at 'energy_max' when
            `per_decade=True`. If True, the number of bins per decade
            might be slightly increased to match the bounds. If False,
            'energy_max' might be reduced so the number of bins per
            decade is exactly the given input. Default is True.

        Returns
        -------
        axis : `MapAxis`
            Create MapAxis from the given input parameters.
        """
        energy_min = u.Quantity(energy_min, unit)
        energy_max = u.Quantity(energy_max, unit)

        if unit is None:
            unit = energy_max.unit
            energy_min = energy_min.to(unit)

        if not energy_max.unit.is_equivalent("TeV"):
            raise ValueError(
                f"Please provide a valid energy unit, got {energy_max.unit} instead."
            )

        if per_decade:
            if strict_bounds:
                nbin = np.ceil(np.log10(energy_max / energy_min).value * nbin)
            else:
                bin_per_decade = nbin
                nbin = np.floor(
                    np.log10(energy_max / energy_min).value * bin_per_decade
                )
                if np.log10(energy_max / energy_min).value % (1 / bin_per_decade) != 0:
                    energy_max = energy_min * 10 ** (nbin / bin_per_decade)

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
        # TODO: What to do with interp in docstring but not in signature?
        """Generate an axis object from a sequence of nodes (bin centers).

        This will create a sequence of bins with edges half-way
        between the node values. This method should be used to
        construct an axis where the bin center should lie at a
        specific value (e.g. a map of a continuous function).

        Parameters
        ----------
        nodes : `~numpy.ndarray`
            Axis nodes (bin center).
        interp : {'lin', 'log', 'sqrt'}
            Interpolation method used to transform between axis and pixel
            coordinates. Default is 'lin'.
        **kwargs : dict, optional
            Keyword arguments passed to `MapAxis`.
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
        **kwargs : dict, optional
            Keyword arguments passed to `MapAxis`.
        """
        if len(edges) < 2:
            raise ValueError("Edges array must have at least two elements.")

        return cls(edges, node_type="edges", **kwargs)

    def concatenate(self, axis):
        """Concatenate another `MapAxis` to this `MapAxis` into a new `MapAxis` object.

        Name, interp type and node type must agree between the axes. If the node
        type is "edges", the edges must be contiguous and non-overlapping.

        Parameters
        ----------
        axis : `MapAxis`
            Axis to concatenate with.

        Returns
        -------
        axis : `MapAxis`
            Concatenation of the two axis.
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

    def pad(self, pad_width):
        """Pad the axis by a given number of pixels.

        Parameters
        ----------
        pad_width : int or tuple of int
            A single integer pads in both direction of the axis, a tuple specifies
            which number of bins to pad at the low and high edge of the axis.

        Returns
        -------
        axis : `MapAxis`
            Padded axis.
        """
        if isinstance(pad_width, tuple):
            pad_low, pad_high = pad_width
        else:
            pad_low, pad_high = pad_width, pad_width

        if self.node_type == "edges":
            pix = np.arange(-pad_low, self.nbin + pad_high + 1) - 0.5
            edges = self.pix_to_coord(pix)
            return self.from_edges(edges=edges, interp=self.interp, name=self.name)
        else:
            pix = np.arange(-pad_low, self.nbin + pad_high)
            nodes = self.pix_to_coord(pix)
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
            Merged axis.
        """
        ax_stacked = axes[0]

        for ax in axes[1:]:
            ax_stacked = ax_stacked.concatenate(ax)

        return ax_stacked

    def pix_to_coord(self, pix):
        """Transform pixel to axis coordinates.

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
        values = self._transform.pix_to_coord(pix=pix)
        return u.Quantity(values, unit=self.unit, copy=False)

    def pix_to_idx(self, pix, clip=False):
        """Convert pixel to index.

        Parameters
        ----------
        pix : `~numpy.ndarray`
            Pixel coordinates.
        clip : bool, optional
            Choose whether to clip indices to the valid range of the
            axis. Default is False. If False, indices for coordinates outside
            the axis range will be set to -1.

        Returns
        -------
        idx : `~numpy.ndarray`
            Pixel indices.
        """
        if clip:
            idx = np.clip(pix, 0, self.nbin - 1)
        else:
            condition = (pix < 0) | (pix >= self.nbin)
            idx = np.where(condition, -1, pix)

        return idx

    def coord_to_pix(self, coord):
        """Transform axis to pixel coordinates.

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
        pix = self._transform.coord_to_pix(coord=coord)
        return np.array(pix + self._pix_offset, ndmin=1)

    def coord_to_idx(self, coord, clip=False):
        """Transform axis coordinate to bin index.

        Parameters
        ----------
        coord : `~numpy.ndarray`
            Array of axis coordinate values.
        clip : bool, optional
            Choose whether to clip the index to the valid range of the
            axis. Default is False. If False, then indices for values outside the axis
            range will be set to -1.

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
            Slice object selecting a sub-selection of the axis.

        Returns
        -------
        axis : `MapAxis`
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
            Squashed axis object.
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
        """Init map axis instance by copying missing init arguments from self."""
        argnames = inspect.getfullargspec(self.__init__).args
        argnames.remove("self")

        for arg in argnames:
            value = getattr(self, "_" + arg)
            if arg not in kwargs:
                kwargs[arg] = copy.deepcopy(value)

        return self.__class__(**kwargs)

    def copy(self, **kwargs):
        """Copy `MapAxis` instance and overwrite given attributes.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments to overwrite in the map axis constructor.

        Returns
        -------
        copy : `MapAxis`
            Copied map axis.
        """
        return self._init_copy(**kwargs)

    def round(self, coord, clip=False):
        """Round coordinate to the nearest axis edge.

        Parameters
        ----------
        coord : `~astropy.units.Quantity`
            Coordinates to be rounded.
        clip : bool, optional
            Choose whether to clip the index to the valid range of the
            axis. Default is False. If False, then indices for values outside the axis
            range will be set to -1.

        Returns
        -------
        coord : `~astropy.units.Quantity`
            Rounded coordinates.
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
        if self.node_type != "edges":
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

        When upsampling for each node specified in the axis, the corresponding
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
            Upsampled map axis.

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

        When downsampling each n-th (given by the factor) bin is selected from
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

    def to_header(self, format="ogip", idx=0):
        """Create FITS header.

        Parameters
        ----------
        format : {"ogip"}, optional
            Format specification. Default is "ogip".
        idx : int, optional
            Column index of the axis. Default is 0.

        Returns
        -------
        header : `~astropy.io.fits.Header`
            Header to extend.
        """
        header = fits.Header()

        if format in ["ogip", "ogip-sherpa"]:
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
        elif format in ["gadf", "fgst-ccube", "fgst-template"]:
            key = f"AXCOLS{idx}"
            name = self.name.upper()

            if self.name == "energy" and self.node_type == "edges":
                header[key] = "E_MIN,E_MAX"
            elif self.name == "energy" and self.node_type == "center":
                header[key] = "ENERGY"
            elif self.node_type == "edges":
                header[key] = f"{name}_MIN,{name}_MAX"
            elif self.node_type == "center":
                header[key] = name
            else:
                raise ValueError(f"Invalid node type {self.node_type!r}")

            key_interp = f"INTERP{idx}"
            header[key_interp] = self.interp

        else:
            raise ValueError(f"Unknown format {format}")

        return header

    def to_table(self, format="ogip"):
        """Convert `~astropy.units.Quantity` to OGIP ``EBOUNDS`` extension.

        See https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html#tth_sEc3.2  # noqa: E501

        The 'ogip-sherpa' format is equivalent to 'ogip' but uses keV energy units.

        Parameters
        ----------
        format : {"ogip", "ogip-sherpa", "gadf-dl3", "gtpsf"}
            Format specification. Default is "ogip".

        Returns
        -------
        table : `~astropy.table.Table`
            Table HDU.
        """
        table = Table()
        edges = self.edges

        if format in ["ogip", "ogip-sherpa"]:
            self.assert_name("energy")

            if format == "ogip-sherpa":
                edges = edges.to("keV")

            table["CHANNEL"] = np.arange(self.nbin, dtype=np.int16)
            table["E_MIN"] = edges[:-1]
            table["E_MAX"] = edges[1:]
        elif format in ["ogip-arf", "ogip-arf-sherpa"]:
            self.assert_name("energy_true")

            if format == "ogip-arf-sherpa":
                edges = edges.to("keV")

            table["ENERG_LO"] = edges[:-1]
            table["ENERG_HI"] = edges[1:]
        elif format == "gadf-sed":
            if self.is_energy_axis:
                table["e_ref"] = self.center
                table["e_min"] = self.edges_min
                table["e_max"] = self.edges_max
        elif format == "gadf-dl3":
            from gammapy.irf.io import IRF_DL3_AXES_SPECIFICATION

            if self.name == "energy":
                column_prefix = "ENERG"
            else:
                for column_prefix, spec in IRF_DL3_AXES_SPECIFICATION.items():
                    if spec["name"] == self.name:
                        break

            if self.node_type == "edges":
                edges_hi, edges_lo = edges[:-1], edges[1:]
            else:
                edges_hi, edges_lo = self.center, self.center

            table[f"{column_prefix}_LO"] = edges_hi[np.newaxis]
            table[f"{column_prefix}_HI"] = edges_lo[np.newaxis]
        elif format == "gtpsf":
            if self.name == "energy_true":
                table["Energy"] = self.center.to("MeV")
            elif self.name == "rad":
                table["Theta"] = self.center.to("deg")
            else:
                raise ValueError(
                    "Can only convert true energy or rad axis to"
                    f"'gtpsf' format, got {self.name}"
                )
        else:
            raise ValueError(f"{format} is not a valid format")

        return table

    def to_table_hdu(self, format="ogip"):
        """Convert `~astropy.units.Quantity` to OGIP ``EBOUNDS`` extension.

        See https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html#tth_sEc3.2  # noqa: E501

        The 'ogip-sherpa' format is equivalent to 'ogip' but uses keV energy units.

        Parameters
        ----------
        format : {"ogip", "ogip-sherpa", "gtpsf"}
            Format specification. Default is "ogip".

        Returns
        -------
        hdu : `~astropy.io.fits.BinTableHDU`
            Table HDU.
        """
        table = self.to_table(format=format)

        if format == "gtpsf":
            name = "THETA"
        else:
            name = None

        hdu = fits.BinTableHDU(table, name=name)

        if format in ["ogip", "ogip-sherpa"]:
            hdu.header.update(self.to_header(format=format))

        return hdu

    @classmethod
    def from_table(cls, table, format="ogip", idx=0, column_prefix=""):
        """Instantiate MapAxis from a table HDU.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table.
        format : {"ogip", "ogip-arf", "fgst-ccube", "fgst-template", "gadf", "gadf-dl3"}
            Format specification. Default is "ogip".
        idx : int, optional
            Column index of the axis. Default is 0.
        column_prefix : str, optional
            Column name prefix of the axis, used for creating the axis. Default is "".

        Returns
        -------
        axis : `MapAxis`
            Map Axis.
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
            hduclass = table.meta.get("HDUCLAS2")
            if hduclass in {"BKG", "RAD_MAX"} and column_prefix == "ENERG":
                name = "energy"

            edges_lo = table[f"{column_prefix}_LO"].quantity[0]
            edges_hi = table[f"{column_prefix}_HI"].quantity[0]
            # for a single-valued array, it can happen that the value is stored/extracted as a scalar
            if edges_lo.isscalar:
                log.warning(
                    f"'{column_prefix}' axis is stored as a scalar -- converting to 1D array."
                )
                edges_lo = edges_lo[np.newaxis]
                edges_hi = edges_hi[np.newaxis]

            if np.allclose(edges_hi, edges_lo):
                axis = MapAxis.from_nodes(edges_hi, interp=interp, name=name)
            else:
                edges = edges_from_lo_hi(edges_lo, edges_hi)
                axis = MapAxis.from_edges(edges, interp=interp, name=name)
        elif format == "gtpsf":
            try:
                energy = table["Energy"].data * u.MeV
                axis = MapAxis.from_nodes(energy, name="energy_true", interp="log")
            except KeyError:
                rad = table["Theta"].data * u.deg
                axis = MapAxis.from_nodes(rad, name="rad")
        elif format == "gadf-sed-energy":
            if "e_min" in table.colnames and "e_max" in table.colnames:
                e_min = flat_if_equal(table["e_min"].quantity)
                e_max = flat_if_equal(table["e_max"].quantity)
                edges = edges_from_lo_hi(e_min, e_max)
                axis = MapAxis.from_energy_edges(edges)
            elif "e_ref" in table.colnames:
                e_ref = flat_if_equal(table["e_ref"].quantity)
                axis = MapAxis.from_nodes(e_ref, name="energy", interp="log")
            else:
                raise ValueError(
                    "Either 'e_ref', 'e_min' or 'e_max' column " "names are required"
                )
        elif format == "gadf-sed-norm":
            # TODO: guess interp here
            nodes = flat_if_equal(table["norm_scan"][0])
            axis = MapAxis.from_nodes(nodes, name="norm")
        elif format == "gadf-sed-counts":
            if "datasets" in table.colnames:
                labels = np.unique(table["datasets"])
                axis = LabelMapAxis(labels=labels, name="dataset")
            else:
                shape = table["counts"].shape
                edges = np.arange(shape[-1] + 1) - 0.5
                axis = MapAxis.from_edges(edges, name="dataset")
        elif format == "profile":
            if "datasets" in table.colnames:
                labels = np.unique(table["datasets"])
                axis = LabelMapAxis(labels=labels, name="dataset")
            else:
                x_ref = table["x_ref"].quantity
                axis = MapAxis.from_nodes(x_ref, name="projected-distance")
        else:
            raise ValueError(f"Format '{format}' not supported")

        return axis

    @classmethod
    def from_table_hdu(cls, hdu, format="ogip", idx=0):
        """Instantiate MapAxis from table HDU.

        Parameters
        ----------
        hdu : `~astropy.io.fits.BinTableHDU`
            Table HDU.
        format : {"ogip", "ogip-arf", "fgst-ccube", "fgst-template"}
            Format specification. Default is "ogip".
        idx : int, optional
            Column index of the axis. Default is 0.

        Returns
        -------
        axis : `MapAxis`
            Map Axis.
        """
        table = Table.read(hdu)
        return cls.from_table(table, format=format, idx=idx)


class MapAxes(Sequence):
    """MapAxis container class.

    Parameters
    ----------
    axes : list of `MapAxis`
        List of map axis objects.
    """

    def __init__(self, axes, n_spatial_axes=None):
        unique_names = []

        for ax in axes:
            if ax.name in unique_names:
                raise (
                    ValueError(f"Axis names must be unique, got: '{ax.name}' twice.")
                )
            unique_names.append(ax.name)

        self._axes = axes
        self._n_spatial_axes = n_spatial_axes

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    @property
    def primary_axis(self):
        """Primary extra axis, defined as the longest one.

        Returns
        -------
        axis : `MapAxis`
            Map axis.
        """
        # get longest axis
        idx = np.argmax(self.shape)
        return self[int(idx)]

    @property
    def is_flat(self):
        """Whether axes is flat."""
        shape = np.array(self.shape)
        return np.all(shape == 1)

    @property
    def is_unidimensional(self):
        """Whether axes is unidimensional."""
        shape = np.array(self.shape)
        non_zero = np.count_nonzero(shape > 1)
        return self.is_flat or non_zero == 1

    @property
    def reverse(self):
        """Reverse axes order."""
        return MapAxes(self[::-1])

    @property
    def iter_with_reshape(self):
        # TODO: The name is misleading. Maybe iter_axis_and_shape?
        """Generator that iterates over axes and their shape."""
        for idx, axis in enumerate(self):
            # Extract values for each axis, default: nodes
            shape = [1] * len(self)
            shape[idx] = -1
            if self._n_spatial_axes:
                shape = (
                    shape[::-1]
                    + [
                        1,
                    ]
                    * self._n_spatial_axes
                )
            yield tuple(shape), axis

    def get_coord(self, mode="center", axis_name=None):
        """Get axes coordinates.

        Parameters
        ----------
        mode : {"center", "edges"}
            Coordinate center or edges. Default is "center".
        axis_name : str, optional
            Axis name for which mode='edges' applies. Default is None.

        Returns
        -------
        coords : dict of `~astropy.units.Quantity`.
            Map coordinates as a dictionary.
        """
        coords = {}

        for shape, axis in self.iter_with_reshape:
            if mode == "edges" and axis.name == axis_name:
                coord = axis.edges
            else:
                coord = axis.center
            coords[axis.name] = coord.reshape(shape)

        return coords

    def bin_volume(self):
        """Bin axes volume.

        Returns
        -------
        bin_volume : `~astropy.units.Quantity`
            Bin volume.
        """
        bin_volume = np.array(1)

        for shape, axis in self.iter_with_reshape:
            bin_volume = bin_volume * axis.bin_width.reshape(shape)

        return bin_volume

    @property
    def shape(self):
        """Shapes of the axes."""
        return tuple([ax.nbin for ax in self])

    @property
    def names(self):
        """Names of the axes."""
        return [ax.name for ax in self]

    def index(self, axis_name):
        """Get index in list."""
        return self.names.index(axis_name)

    def index_data(self, axis_name):
        """Get data index of the axes.

        Parameters
        ----------
        axis_name : str
            Name of the axis.

        Returns
        -------
        idx : int
            Data index.
        """
        idx = self.names.index(axis_name)
        return len(self) - idx - 1

    def __len__(self):
        return len(self._axes)

    def __add__(self, other):
        return self.__class__(list(self) + list(other))

    def upsample(self, factor, axis_name):
        """Upsample axis by a given factor.

        Parameters
        ----------
        factor : int
            Upsampling factor.
        axis_name : str
            Axis to upsample.

        Returns
        -------
        axes : `MapAxes`
            Map axes.
        """
        axes = []

        for ax in self:
            if ax.name == axis_name:
                ax = ax.upsample(factor=factor)

            axes.append(ax.copy())

        return self.__class__(axes=axes)

    def replace(self, axis):
        """Replace a given axis. In order to be replaced,
        the name of the new axis must match the name of the old axis.

        Parameters
        ----------
        axis : `MapAxis`
            Map axis.

        Returns
        -------
        axes : MapAxes
            Map axes.
        """
        axes = []

        for ax in self:
            if ax.name == axis.name:
                ax = axis

            axes.append(ax)

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
            groups[axis.name + "_min"].quantity,
            groups[axis.name + "_max"].quantity,
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
        """Downsample axis by a given factor.

        Parameters
        ----------
        factor : int
            Downsampling factor.
        axis_name : str
            Axis to downsample.

        Returns
        -------
        axes : `MapAxes`
            Map axes.

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

    def pad(self, axis_name, pad_width):
        """Pad axis.

        Parameters
        ----------
        axis_name : str
            Name of the axis to pad.
        pad_width : int or tuple of int
            Pad width.

        Returns
        -------
        axes : `MapAxes`
            Axes with squashed axis.
        """
        axes = []

        for ax in self:
            if ax.name == axis_name:
                ax = ax.pad(pad_width=pad_width)
            axes.append(ax)

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
            Axes without the `axis_name`.
        """
        axes = []
        for ax in self:
            if ax.name == axis_name:
                continue
            axes.append(ax.copy())

        return self.__class__(axes=axes)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._axes[idx]
        elif isinstance(idx, str):
            for ax in self._axes:
                if ax.name == idx:
                    return ax
            raise KeyError(f"No axes: {idx!r}")
        elif isinstance(idx, slice):
            axes = self._axes[idx]
            return self.__class__(axes=axes)
        elif isinstance(idx, list):
            axes = []
            for name in idx:
                axes.append(self[name])

            return self.__class__(axes=axes)
        else:
            raise TypeError(f"Invalid type: {type(idx)!r}")

    def coord_to_idx(self, coord, clip=True):
        """Transform from axis to pixel indices.

        Parameters
        ----------
        coord : dict of `~numpy.ndarray` or `MapCoord`
            Array of axis coordinate values.
        clip : bool, optional
            Choose whether to clip indices to the valid range of the axis. Default is True.
            If False, then indices for coordinates outside the axis range will be set to -1.


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

    def pix_to_idx(self, pix, clip=False):
        """Convert pixel to pixel indices.

        Parameters
        ----------
        pix : tuple of `~numpy.ndarray`
            Pixel coordinates.
        clip : bool, optional
            Choose whether to clip indices to the valid range of the
            axis. Default is False. If False, then indices for coordinates outside
            the axi range will be set to -1.

        Returns
        -------
        idx : tuple `~numpy.ndarray`
            Pixel indices.
        """
        idx = []

        for pix_array, ax in zip(pix, self):
            idx.append(ax.pix_to_idx(pix_array, clip=clip))

        return tuple(idx)

    def slice_by_idx(self, slices):
        """Create a new geometry by slicing the non-spatial axes.

        Parameters
        ----------
        slices : dict
            Dictionary of axes names and integers or `slice` object pairs. Contains one
            element for each non-spatial dimension. For integer indexing the
            corresponding axes is dropped from the map. Axes not specified in the
            dictionary are kept unchanged.

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

    def to_header(self, format="gadf"):
        """Convert axes to FITS header.

        Parameters
        ----------
        format : {"gadf"}
            Header format. Default is "gadf".

        Returns
        -------
        header : `~astropy.io.fits.Header`
            FITS header.
        """
        header = fits.Header()

        for idx, ax in enumerate(self, start=1):
            header_ax = ax.to_header(format=format, idx=idx)
            header.update(header_ax)

        return header

    def to_table(self, format="gadf"):
        """Convert axes to table.

        Parameters
        ----------
        format : {"gadf", "gadf-dl3", "fgst-ccube", "fgst-template", "ogip", "ogip-sherpa", "ogip-arf", "ogip-arf-sherpa"}  # noqa E501
            Format to use. Default is "gadf".

        Returns
        -------
        table : `~astropy.table.Table`
            Table with axis data.
        """
        if format == "gadf-dl3":
            tables = []

            for ax in self:
                tables.append(ax.to_table(format=format))

            table = hstack(tables)
        elif format in ["gadf", "fgst-ccube", "fgst-template"]:
            table = Table()
            table["CHANNEL"] = np.arange(np.prod(self.shape))

            axes_ctr = np.meshgrid(*[ax.center for ax in self])
            axes_min = np.meshgrid(*[ax.edges_min for ax in self])
            axes_max = np.meshgrid(*[ax.edges_max for ax in self])

            for idx, ax in enumerate(self):
                name = ax.name.upper()

                if name == "ENERGY":
                    colnames = ["ENERGY", "E_MIN", "E_MAX"]
                else:
                    colnames = [name, name + "_MIN", name + "_MAX"]

                for colname, v in zip(colnames, [axes_ctr, axes_min, axes_max]):
                    # do not store edges for label axis
                    if ax.node_type == "label" and colname != name:
                        continue

                    table[colname] = np.ravel(v[idx])

                if isinstance(ax, TimeMapAxis):
                    ref_dict = time_ref_to_dict(ax.reference_time)
                    table.meta.update(ref_dict)

        elif format in ["ogip", "ogip-sherpa", "ogip", "ogip-arf"]:
            energy_axis = self["energy"]
            table = energy_axis.to_table(format=format)
        else:
            raise ValueError(f"Unsupported format: '{format}'")

        return table

    def to_table_hdu(self, format="gadf", hdu_bands=None):
        """Make FITS table columns for map axes.

        Parameters
        ----------
        format : {"gadf", "fgst-ccube", "fgst-template"}
            Format to use. Default is "gadf".
        hdu_bands : str, optional
            Name of the bands HDU to use. Default is None.

        Returns
        -------
        hdu : `~astropy.io.fits.BinTableHDU`
            Bin table HDU.
        """
        # FIXME: Check whether convention is compatible with
        #  dimensionality of geometry and simplify!!!

        if format in ["fgst-ccube", "ogip", "ogip-sherpa"]:
            hdu_bands = "EBOUNDS"
        elif format == "fgst-template":
            hdu_bands = "ENERGIES"
        elif format == "gadf" or format is None:
            if hdu_bands is None:
                hdu_bands = "BANDS"
        else:
            raise ValueError(f"Unknown format {format}")

        table = self.to_table(format=format)
        header = self.to_header(format=format)
        return fits.BinTableHDU(table, name=hdu_bands, header=header)

    @classmethod
    def from_table_hdu(cls, hdu, format="gadf"):
        """Create MapAxes from BinTableHDU.

        Parameters
        ----------
        hdu : `~astropy.io.fits.BinTableHDU`
            Bin table HDU.
        format : {"gadf", "gadf-dl3", "fgst-ccube", "fgst-template", "fgst-bexcube", "ogip-arf"}
            Format to use. Default is "gadf".


        Returns
        -------
        axes : `MapAxes`
            Map axes object.
        """
        if hdu is None:
            return cls([])

        table = Table.read(hdu)
        return cls.from_table(table, format=format)

    @classmethod
    def from_table(cls, table, format="gadf"):
        """Create MapAxes from table.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Bin table HDU.
        format : {"gadf", "gadf-dl3", "fgst-ccube", "fgst-template", "fgst-bexcube", "ogip-arf"}
            Format to use. Default is "gadf".

        Returns
        -------
        axes : `MapAxes`
            Map axes object.
        """
        from gammapy.irf.io import IRF_DL3_AXES_SPECIFICATION

        axes = []

        # Formats that support only one energy axis
        if format in [
            "fgst-ccube",
            "fgst-template",
            "fgst-bexpcube",
            "ogip",
            "ogip-arf",
        ]:
            axes.append(MapAxis.from_table(table, format=format))
        elif format == "gadf":
            # This limits the max number of axes to 5
            for idx in range(5):
                axcols = table.meta.get("AXCOLS{}".format(idx + 1))
                if axcols is None:
                    break

                # TODO: what is good way to check whether it is a given axis type?
                try:
                    axis = LabelMapAxis.from_table(table, format=format, idx=idx)
                except (KeyError, TypeError):
                    try:
                        axis = TimeMapAxis.from_table(table, format=format, idx=idx)
                    except (KeyError, ValueError, IndexError):
                        axis = MapAxis.from_table(table, format=format, idx=idx)

                axes.append(axis)
        elif format == "gadf-dl3":
            for column_prefix in IRF_DL3_AXES_SPECIFICATION:
                try:
                    axis = MapAxis.from_table(
                        table, format=format, column_prefix=column_prefix
                    )
                except KeyError:
                    continue

                axes.append(axis)
        elif format == "gadf-sed":
            for axis_format in ["gadf-sed-norm", "gadf-sed-energy", "gadf-sed-counts"]:
                try:
                    axis = MapAxis.from_table(table=table, format=axis_format)
                except KeyError:
                    continue
                axes.append(axis)
        elif format == "lightcurve":
            axes.extend(cls.from_table(table=table, format="gadf-sed"))
            axes.append(TimeMapAxis.from_table(table, format="lightcurve"))
        elif format == "profile":
            axes.extend(cls.from_table(table=table, format="gadf-sed"))
            axes.append(MapAxis.from_table(table, format="profile"))
        else:
            raise ValueError(f"Unsupported format: '{format}'")

        return cls(axes)

    @classmethod
    def from_default(cls, axes, n_spatial_axes=None):
        """Make a sequence of `~MapAxis` objects.

        Parameters
        ----------
        axes : list of `~MapAxis` or `~numpy.ndarray`
            Sequence of axis or edges defining the axes.
        n_spatial_axes : int, optional
            Number of spatial axes. Default is None.

        Returns
        -------
        axes : `MapAxes`
            Map axes object.
        """
        if axes is None:
            return cls([])

        axes_out = []
        for idx, ax in enumerate(axes):
            if isinstance(ax, np.ndarray):
                ax = MapAxis(ax)

            if ax.name == "":
                ax._name = f"axis{idx}"

            axes_out.append(ax)

        return cls(axes_out, n_spatial_axes=n_spatial_axes)

    def assert_names(self, required_names):
        """Assert required axis names and order.

        Parameters
        ----------
        required_names : list of str
            Required names.
        """
        message = (
            "Incorrect axis order or names. Expected axis "
            f"order: {required_names}, got: {self.names}."
        )

        if not len(self) == len(required_names):
            raise ValueError(message)

        try:
            for ax, required_name in zip(self, required_names):
                ax.assert_name(required_name)

        except ValueError:
            raise ValueError(message)

    def rename_axes(self, names, new_names):
        """Rename the axes.

        Parameters
        ----------
        names : str or list of str
            Names of the axis.
        new_names : str or list of str
            New names of the axes (list must be of same length than `names`).

        Returns
        -------
        axes : `MapAxes`
            Renamed Map axes object.
        """
        axes = self.copy()
        if isinstance(names, str):
            names = [names]
        if isinstance(new_names, str):
            new_names = [new_names]
        for name, new_name in zip(names, new_names):
            axes[name]._name = new_name
        return axes

    @property
    def center_coord(self):
        """Center coordinates."""
        return tuple([ax.pix_to_coord((float(ax.nbin) - 1.0) / 2.0) for ax in self])

    def is_allclose(self, other, **kwargs):
        """Check if other map axes are all close.

        Parameters
        ----------
        other : `MapAxes`
            Other map axes.
        **kwargs : dict, optional
            Keyword arguments forwarded to `~MapAxis.is_allclose`

        Returns
        -------
        is_allclose : bool
            Whether other axes are all close.
        """
        if not isinstance(other, self.__class__):
            return TypeError(f"Cannot compare {type(self)} and {type(other)}")

        return np.all([ax0.is_allclose(ax1, **kwargs) for ax0, ax1 in zip(other, self)])

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.is_allclose(other, rtol=1e-6, atol=1e-6)

    def __ne__(self, other):
        return not self.__eq__(other)

    def copy(self):
        """Initialize a new map axes instance by copying each axis."""
        return self.__class__([_.copy() for _ in self])


class TimeMapAxis:
    """Class representing a time axis.

    Provides methods for transforming to/from axis and pixel coordinates.
    A time axis can represent non-contiguous sequences of non-overlapping time intervals.

    Time intervals must be provided in increasing order.

    Parameters
    ----------
    edges_min : `~astropy.units.Quantity`
        Array of edge time values. This is the time delta w.r.t. to the reference time.
    edges_max : `~astropy.units.Quantity`
        Array of edge time values. This is the time delta w.r.t. to the reference time.
    reference_time : `~astropy.time.Time`
        Reference time to use.
    name : str, optional
        Axis name. Default is "time".
    interp : {'lin'}
        Interpolation method used to transform between axis and pixel
        coordinates. For now only 'lin' is supported. Default is 'lin'.
    """

    node_type = "intervals"

    def __init__(self, edges_min, edges_max, reference_time, name="time", interp="lin"):
        self._name = name
        self._time_format = "iso"

        edges_min = u.Quantity(edges_min, ndmin=1)
        edges_max = u.Quantity(edges_max, ndmin=1)

        if not edges_min.unit.is_equivalent("s"):
            raise ValueError(
                f"Time edges min must have a valid time unit, got {edges_min.unit}"
            )

        if not edges_max.unit.is_equivalent("s"):
            raise ValueError(
                f"Time edges max must have a valid time unit, got {edges_max.unit}"
            )

        if not edges_min.shape == edges_max.shape:
            raise ValueError(
                "Edges min and edges max must have the same shape,"
                f" got {edges_min.shape} and {edges_max.shape}."
            )

        if not np.all(edges_max > edges_min):
            raise ValueError("Edges max must all be larger than edge min")

        if not np.all(edges_min == np.sort(edges_min)):
            raise ValueError("Time edges min values must be sorted")

        if not np.all(edges_max == np.sort(edges_max)):
            raise ValueError("Time edges max values must be sorted")

        if interp != "lin":
            raise NotImplementedError(
                f"Non-linear scaling scheme are not supported yet, got {interp}"
            )

        self._edges_min = edges_min
        self._edges_max = edges_max
        self._reference_time = Time(reference_time)
        self._pix_offset = -0.5
        self._interp = interp

        delta = edges_min[1:] - edges_max[:-1]
        if np.any(delta < 0 * u.s):
            raise ValueError("Time intervals must not overlap.")

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    @property
    def is_contiguous(self):
        """Whether the axis is contiguous."""
        return np.all(self.edges_min[1:] == self.edges_max[:-1])

    def to_contiguous(self):
        """Make the time axis contiguous.

        Returns
        -------
        axis : `TimeMapAxis`
            Contiguous time axis.
        """
        edges = np.unique(np.stack([self.edges_min, self.edges_max]))
        return self.__class__(
            edges_min=edges[:-1],
            edges_max=edges[1:],
            reference_time=self.reference_time,
            name=self.name,
            interp=self.interp,
        )

    @property
    def unit(self):
        """Axis unit."""
        return self.edges_max.unit

    @property
    def interp(self):
        """Interpolation scale of the axis."""
        return self._interp

    @property
    def reference_time(self):
        """Return reference time used for the axis."""
        return self._reference_time

    @property
    def name(self):
        """Return the axis name."""
        return self._name

    @property
    def nbin(self):
        """Return the number of bins in the axis."""
        return len(self.edges_min.flatten())

    @property
    def edges_min(self):
        """Return the array of bin edges maximum values."""
        return self._edges_min

    @property
    def edges_max(self):
        """Return the array of bin edges minimum values."""
        return self._edges_max

    @property
    def edges(self):
        """Return the array of bin edges values."""
        if not self.is_contiguous:
            raise ValueError("Time axis is not contiguous")

        return edges_from_lo_hi(self.edges_min, self.edges_max)

    @property
    def bounds(self):
        """Bounds of the axis as a ~astropy.units.Quantity."""
        return self.edges_min[0], self.edges_max[-1]

    @property
    def time_bounds(self):
        """Bounds of the axis as a ~astropy.units.Quantity."""
        t_min, t_max = self.bounds
        return t_min + self.reference_time, t_max + self.reference_time

    @property
    def time_min(self):
        """Return axis lower edges as `~astropy.time.Time` object."""
        return self._edges_min + self.reference_time

    @property
    def time_max(self):
        """Return axis upper edges as a `~astropy.time.Time` object."""
        return self._edges_max + self.reference_time

    @property
    def time_delta(self):
        """Return axis time bin width as a `~astropy.time.TimeDelta` object."""
        return self.time_max - self.time_min

    @property
    def time_mid(self):
        """Return time bin center as a `~astropy.time.Time` object."""
        return self.time_min + 0.5 * self.time_delta

    @property
    def time_edges(self):
        """Time edges as a `~astropy.time.Time` object."""
        return self.reference_time + self.edges

    @property
    def time_format(self):
        # TODO: The docsttring for time format and its setter are not showing up in the docs.
        """The time format to use for the axis."""
        return self._time_format

    @time_format.setter
    def time_format(self, val):
        """The time format to use for the axis."""
        if val not in ["iso", "mjd"]:
            raise ValueError(f"Invalid time_format: {self.time_format}")
        self._time_format = val

    @property
    def as_plot_xerr(self):
        """Return x errors for plotting."""
        xn, xp = self.time_mid - self.time_min, self.time_max - self.time_mid

        if self.time_format == "iso":
            x_errn = xn.to_datetime()
            x_errp = xp.to_datetime()
        else:
            x_errn = xn.to("day")
            x_errp = xp.to("day")

        return x_errn, x_errp

    @property
    def as_plot_labels(self):
        """Return labels for plotting."""
        labels = []

        for t_min, t_max in self.iter_by_edges:
            label = f"{getattr(t_min, self.time_format)} - {getattr(t_max, self.time_format)}"
            labels.append(label)

        return labels

    @property
    def as_plot_edges(self):
        """Return edges for plotting."""
        if self.time_format == "iso":
            edges = self.time_edges.to_datetime()
        else:
            edges = self.time_edges.mjd * u.day

        return edges

    @property
    def as_plot_center(self):
        """Return center for plotting."""
        if self.time_format == "iso":
            center = self.time_mid.datetime
        else:
            center = self.time_mid.mjd * u.day
        return center

    def format_plot_xaxis(self, ax):
        """Format the x-axis.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axis`
            Plot axis to format.

        Returns
        -------
        ax : `~matplotlib.pyplot.Axis`
            Formatted plot axis.
        """
        from matplotlib.dates import DateFormatter

        xlabel = DEFAULT_LABEL_TEMPLATE.format(
            quantity=PLOT_AXIS_LABEL.get(self.name, self.name.capitalize()),
            unit=self.time_format,
        )
        ax.set_xlabel(xlabel)

        if self.time_format == "iso":
            ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M:%S"))
        else:
            ax.xaxis.set_major_formatter("{x:,.5f}")
        plt.setp(
            ax.xaxis.get_majorticklabels(),
            rotation=30,
            ha="right",
            rotation_mode="anchor",
        )
        return ax

    def assert_name(self, required_name):
        """Assert axis name if a specific one is required.

        Parameters
        ----------
        required_name : str
            Required name.
        """
        if self.name != required_name:
            raise ValueError(
                "Unexpected axis name,"
                f' expected "{required_name}", got: "{self.name}"'
            )

    def is_allclose(self, other, **kwargs):
        """Check if other time map axis is all close.

        Parameters
        ----------
        other : `TimeMapAxis`
            Other time map axis.
        **kwargs : dict, optional
            Keyword arguments forwarded to `~numpy.allclose`.

        Returns
        -------
        is_allclose : bool
            Whether the other time map axis is allclose.
        """
        if not isinstance(other, self.__class__):
            return TypeError(f"Cannot compare {type(self)} and {type(other)}")

        if self._edges_min.shape != other._edges_min.shape:
            return False

        # This will test equality at microsec level.
        delta_min = self.time_min - other.time_min
        delta_max = self.time_max - other.time_max

        return (
            np.allclose(delta_min.to_value("s"), 0.0, **kwargs)
            and np.allclose(delta_max.to_value("s"), 0.0, **kwargs)
            and self._interp == other._interp
            and self.name.upper() == other.name.upper()
        )

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.is_allclose(other=other, atol=1e-6)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return id(self)

    def is_aligned(self, other, atol=2e-2):
        """Not supported for time axis."""
        raise NotImplementedError

    @property
    def iter_by_edges(self):
        """Iterate by intervals defined by the edges."""
        for time_min, time_max in zip(self.time_min, self.time_max):
            yield (time_min, time_max)

    def coord_to_idx(self, coord, **kwargs):
        """Transform time axis coordinate to bin index.

        Indices of time values falling outside time bins will be
        set to -1.

        Parameters
        ----------
        coord : `~astropy.time.Time` or `~astropy.units.Quantity`
            Array of time axis coordinate values. The quantity is assumed
            to be relative to the reference time.

        Returns
        -------
        idx : `~numpy.ndarray`
            Array of bin indices.
        """
        if isinstance(coord, u.Quantity):
            coord = self.reference_time + coord

        time = Time(coord[..., np.newaxis])
        delta_plus = (time - self.time_min).value > 0.0
        delta_minus = (time - self.time_max).value <= 0.0
        mask = np.logical_and(delta_plus, delta_minus)

        idx = np.asanyarray(np.argmax(mask, axis=-1))
        idx[~np.any(mask, axis=-1)] = INVALID_INDEX.int
        return idx

    def pix_to_coord(self, pix):
        """Transform from pixel position to time coordinate.
        Currently, works only for linear interpolation scheme.

        Parameters
        ----------
        pix : `~numpy.ndarray`
            Array of pixel positions.

        Returns
        -------
        coord : `~astropy.time.Time`
            Array of time axis coordinate values.
        """
        shape = np.shape(pix)
        pix = np.atleast_1d(pix)
        coords = np.zeros_like(pix)
        frac, idx = np.modf(pix)
        idx1 = idx.astype(int)
        valid = np.logical_and(idx >= 0, idx < self.nbin, np.isfinite(idx))
        idx_valid = np.where(valid)
        idx_invalid = np.where(~valid)

        coords[idx_valid] = (
            frac[idx_valid] * self.time_delta[idx1[valid]] + self.edges_min[idx1[valid]]
        ).value
        coords = coords * self.unit + self.reference_time
        coords[idx_invalid] = Time(INVALID_VALUE.time, scale=self.reference_time.scale)
        return coords.reshape(shape)

    def coord_to_pix(self, coord, **kwargs):
        """Transform time axis coordinate to pixel position.

        Pixels of time values falling outside time bins will be
        set to -1.

        Parameters
        ----------
        coord : `~astropy.time.Time`
            Array of time axis coordinate values.

        Returns
        -------
        pix : `~numpy.ndarray`
            Array of pixel positions.
        """
        if isinstance(coord, u.Quantity):
            coord = self.reference_time + coord

        idx = np.atleast_1d(self.coord_to_idx(coord))

        valid_pix = idx != INVALID_INDEX.int
        pix = np.atleast_1d(idx).astype("float")

        # TODO: is there the equivalent of np.atleast1d for astropy.time.Time?
        if coord.shape == ():
            coord = coord.reshape((1,))

        relative_time = coord[valid_pix] - self.reference_time

        scale = interpolation_scale(self._interp)
        valid_idx = idx[valid_pix]
        s_min = scale(self._edges_min[valid_idx])
        s_max = scale(self._edges_max[valid_idx])
        s_coord = scale(relative_time.to(self._edges_min.unit))

        pix[valid_pix] += (s_coord - s_min) / (s_max - s_min)
        pix[~valid_pix] = INVALID_INDEX.float
        return pix - 0.5

    @staticmethod
    def pix_to_idx(pix, clip=False):
        # TODO: Is this useful at all?
        return pix

    @property
    def center(self):
        """Return interval centers as a `~astropy.units.Quantity`."""
        return self.edges_min + 0.5 * self.bin_width

    @property
    def bin_width(self):
        """Return time interval width as a `~astropy.units.Quantity`."""
        return self.time_delta.to("h")

    def __repr__(self):
        str_ = self.__class__.__name__ + "\n"
        str_ += "-" * len(self.__class__.__name__) + "\n\n"
        fmt = "\t{:<14s} : {:<10s}\n"
        str_ += fmt.format("name", self.name)
        str_ += fmt.format("nbins", str(self.nbin))
        str_ += fmt.format("reference time", self.reference_time.iso)
        str_ += fmt.format("scale", self.reference_time.scale)
        str_ += fmt.format("time min.", self.time_min.min().iso)
        str_ += fmt.format("time max.", self.time_max.max().iso)
        str_ += fmt.format("total time", np.sum(self.bin_width))
        return str_.expandtabs(tabsize=2)

    def upsample(self):
        """Not supported for time axis."""
        raise NotImplementedError

    def downsample(self):
        """Not supported for time axis."""
        raise NotImplementedError

    def _init_copy(self, **kwargs):
        """Init map axis instance by copying missing init arguments from self."""
        argnames = inspect.getfullargspec(self.__init__).args
        argnames.remove("self")

        for arg in argnames:
            value = getattr(self, "_" + arg)
            if arg not in kwargs:
                kwargs[arg] = copy.deepcopy(value)

        return self.__class__(**kwargs)

    def copy(self, **kwargs):
        """Copy `TimeMapAxis` instance and overwrite given attributes.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments to overwrite in the map axis constructor.

        Returns
        -------
        copy : `TimeMapAxis`
            Copied time map axis.
        """
        return self._init_copy(**kwargs)

    def slice(self, idx):
        """Create a new axis object by extracting a slice from this axis.

        Parameters
        ----------
        idx : slice
            Slice object selecting a sub-selection of the axis.

        Returns
        -------
        axis : `~TimeMapAxis`
            Sliced time map axis object.
        """
        return TimeMapAxis(
            self._edges_min[idx].copy(),
            self._edges_max[idx].copy(),
            self.reference_time,
            interp=self._interp,
            name=self.name,
        )

    def squash(self):
        """Create a new axis object by squashing the axis into one bin.

        Returns
        -------
        axis : `~MapAxis`
            Squashed time map axis object.
        """
        return TimeMapAxis(
            self._edges_min[0],
            self._edges_max[-1],
            self.reference_time,
            interp=self._interp,
            name=self._name,
        )

    # TODO: if we are to allow log or sqrt bins the reference time should always
    #  be strictly lower than all times
    #  Should we define a mechanism to ensure this is always correct?
    @classmethod
    def from_time_edges(cls, time_min, time_max, unit="d", interp="lin", name="time"):
        """Create TimeMapAxis from the time interval edges defined as a `~astropy.time.Time` object.

        The reference time is defined as the lower edge of the first interval.

        Parameters
        ----------
        time_min : `~astropy.time.Time`
            Array of lower edge times.
        time_max : `~astropy.time.Time`
            Array of lower edge times.
        unit : `~astropy.units.Unit` or str, optional
            The unit to convert the edges to. Default is 'd' (day).
        interp : {'lin'}
            Interpolation method used to transform between axis and pixel
            coordinates. Currently, only 'lin' is supported. Default is 'lin'.
        name : str, optional
            Axis name. Default is "time".

        Returns
        -------
        axis : `TimeMapAxis`
            Time map axis.
        """
        unit = u.Unit(unit)
        reference_time = time_min[0]
        edges_min = time_min - reference_time
        edges_max = time_max - reference_time

        return cls(
            edges_min.to(unit),
            edges_max.to(unit),
            reference_time,
            interp=interp,
            name=name,
        )

    # TODO: how configurable should that be? column names?
    @classmethod
    def from_table(cls, table, format="gadf", idx=0):
        """Create time map axis from table

        Parameters
        ----------
        table : `~astropy.table.Table`
            Bin table HDU.
        format : {"gadf", "fermi-fgl", "lightcurve"}
            Format to use. Default is "gadf".
        idx : int
            Axis index. Default is 0.

        Returns
        -------
        axis : `TimeMapAxis`
            Time map axis.
        """
        if format == "gadf":
            axcols = table.meta.get("AXCOLS{}".format(idx + 1))
            colnames = axcols.split(",")
            name = colnames[0].replace("_MIN", "").lower()
            reference_time = time_ref_from_dict(table.meta)
            edges_min = np.unique(table[colnames[0]].quantity)
            edges_max = np.unique(table[colnames[1]].quantity)
        elif format == "fermi-fgl":
            meta = table.meta.copy()
            meta["MJDREFF"] = str(meta["MJDREFF"]).replace("D-4", "e-4")
            reference_time = time_ref_from_dict(meta=meta)
            name = "time"
            edges_min = table["Hist_Start"][:-1]
            edges_max = table["Hist_Start"][1:]
        elif format == "lightcurve":
            # TODO: is this a good format? It just supports mjd...
            name = "time"
            time_ref_dict = dict(
                MJDREFF=table.meta.get("MJDREFF", 0),
                MJDREFI=table.meta.get("MJDREFI", 0),
                TIMESYS=table.meta.get("TIMESYS", "utc"),
                TIMEUNIT=table.meta.get("TIMEUNIT", "d"),
            )
            reference_time = time_ref_from_dict(time_ref_dict, format="mjd")
            time_min = reference_time + table["time_min"].data * u.Unit(
                time_ref_dict["TIMEUNIT"]
            )
            time_max = reference_time + table["time_max"].data * u.Unit(
                time_ref_dict["TIMEUNIT"]
            )

            if reference_time.mjd == 0:
                # change to a more recent reference time
                reference_time = Time(
                    "2001-01-01T00:00:00", scale=time_ref_dict["TIMESYS"]
                )
                reference_time.format = "mjd"

            edges_min = (time_min - reference_time).to("s")
            edges_max = (time_max - reference_time).to("s")
        else:
            raise ValueError(f"Not a supported format: {format}")

        return cls(
            edges_min=edges_min,
            edges_max=edges_max,
            reference_time=reference_time,
            name=name,
        )

    @classmethod
    def from_gti(cls, gti, name="time"):
        """Create a time axis from an input GTI.

        Parameters
        ----------
        gti : `~gammapy.data.GTI`
            GTI table.
        name : str, optional
            Axis name. Default is "time".

        Returns
        -------
        axis : `TimeMapAxis`
            Time map axis.

        """
        tmin = gti.time_start - gti.time_ref
        tmax = gti.time_stop - gti.time_ref

        return cls(
            edges_min=tmin.to("s"),
            edges_max=tmax.to("s"),
            reference_time=gti.time_ref,
            name=name,
        )

    @classmethod
    def from_gti_bounds(cls, gti, t_delta, name="time"):
        """Create a time axis from an input GTI.

        The unit for the axis is taken from the t_delta quantity.

        Parameters
        ----------
        gti : `~gammapy.data.GTI`
            GTI table.
        t_delta : `~astropy.units.Quantity`
            Time binning.
        name : str, optional
            Axis name. Default is "time".

        Returns
        -------
        axis : `TimeMapAxis`
            Time map axis.

        """
        time_min = gti.time_start[0]
        time_max = gti.time_stop[-1]

        nbin = int(((time_max - time_min) / t_delta).to(""))
        return TimeMapAxis.from_time_bounds(
            time_min=time_min,
            time_max=time_max,
            nbin=nbin,
            name=name,
            unit=t_delta.unit,
        )

    @classmethod
    def from_time_bounds(cls, time_min, time_max, nbin, unit="d", name="time"):
        """Create linearly spaced time axis from bounds.

        Parameters
        ----------
        time_min : `~astropy.time.Time`
            Lower bound.
        time_max : `~astropy.time.Time`
            Upper bound.
        nbin : int
            Number of bins.
        unit : `~astropy.units.Unit` or str, optional
            The unit to convert the edges to. Default is 'd' (day).
        name : str, optional
            Name of the axis. Default is "time".
        """
        delta = time_max - time_min
        time_edges = time_min + delta * np.linspace(0, 1, nbin + 1)
        return cls.from_time_edges(
            time_min=time_edges[:-1],
            time_max=time_edges[1:],
            interp="lin",
            unit=unit,
            name=name,
        )

    def to_gti(self):
        """Convert the axis to a GTI table.

        Returns
        -------
        gti : `~gammapy.data.GTI`
            GTI table.
        """
        from gammapy.data import GTI

        return GTI.create(
            self.edges_min, self.edges_max, reference_time=self.reference_time
        )

    def to_header(self, format="gadf", idx=0):
        """Create FITS header.

        Parameters
        ----------
        format : {"ogip"}
            Format specification. Default is "gadf".
        idx : int, optional
            Column index of the axis. Default is 0.

        Returns
        -------
        header : `~astropy.io.fits.Header`
            Corresponding FITS header.
        """
        header = fits.Header()

        if format == "gadf":
            key = f"AXCOLS{idx}"
            name = self.name.upper()
            header[key] = f"{name}_MIN,{name}_MAX"
            key_interp = f"INTERP{idx}"
            header[key_interp] = self.interp

            ref_dict = time_ref_to_dict(self.reference_time)
            header.update(ref_dict)
        else:
            raise ValueError(f"Unknown format {format}")

        return header

    def group_table(self, interval_edges):
        """Compute bin groups table for the TimeMapAxis, given coarser bin edges.

        Parameters
        ----------
        interval_edges : list of `~astropy.time.Time` or `~astropy.units.Quantity`
            Start and stop time for each interval to compute the LC.

        Returns
        -------
        groups : `~astropy.table.Table`
            Group table. Bin groups are divided in:

             *"normal" for the bins containing data
             *"underflow" for the bins falling below the minimum axis threshold
             *"overflow" for the bins falling above  the maximum axis threshold
             *"outflow" for other states
        """

        for _, edge in enumerate(interval_edges):
            if not isinstance(edge, Time):
                interval_edges[_] = self.reference_time + interval_edges[_]

        time_intervals = list(zip(interval_edges[::2], interval_edges[1::2]))
        group_table = Table(
            names=("idx_min", "idx_max", "time_min", "time_max", "bin_type"),
            dtype=("i8", "i8", "f8", "f8", "S10"),
        )

        for time_interval in time_intervals:
            mask1 = self.time_min >= time_interval[0]
            mask2 = self.time_max <= time_interval[1]
            mask = mask1 & mask2
            if np.any(mask):
                idx_min = np.where(mask)[0][0]
                idx_max = np.where(mask)[0][-1]
                bin_type = "normal   "
            else:
                idx_min = idx_max = -1
                if np.any(mask1):
                    bin_type = "overflow"
                elif np.any(mask2):
                    bin_type = "underflow"
                else:
                    bin_type = "outflow"
            time_min = self.time_min[idx_min]
            time_max = self.time_max[idx_max]
            group_table.add_row(
                [idx_min, idx_max, time_min.mjd, time_max.mjd, bin_type]
            )

        return group_table


class LabelMapAxis:
    """Map axis using labels.

    Parameters
    ----------
    labels : list of str
        Labels to be used for the axis nodes.
    name : str, optional
        Name of the axis. Default is "".

    """

    node_type = "label"

    def __init__(self, labels, name=""):
        unique_labels = np.unique(labels)

        if not len(unique_labels) == len(labels):
            raise ValueError("Node labels must be unique")

        self._labels = unique_labels
        self._name = name

    @property
    def unit(self):
        # TODO: should we allow units for label axis?
        """Unit of the axis."""
        return u.Unit("")

    @property
    def name(self):
        """Name of the axis."""
        return self._name

    def assert_name(self, required_name):
        """Assert axis name if a specific one is required.

        Parameters
        ----------
        required_name : str
            Required name.
        """
        if self.name != required_name:
            raise ValueError(
                "Unexpected axis name,"
                f' expected "{required_name}", got: "{self.name}"'
            )

    @property
    def nbin(self):
        """Number of bins."""
        return len(self._labels)

    def pix_to_coord(self, pix):
        """Transform pixel to label coordinate.

        Parameters
        ----------
        pix : `~numpy.ndarray`
            Array of pixel coordinate values.

        Returns
        -------
        coord : `~numpy.ndarray`
            Array of axis coordinate values.
        """
        idx = np.round(pix).astype(int)
        return self._labels[idx]

    def coord_to_idx(self, coord, **kwargs):
        """Transform label coordinate to indices.

        If the label is not present an error is raised.

        Parameters
        ----------
        coord : `~astropy.time.Time`
            Array of axis coordinate values.

        Returns
        -------
        idx : `~numpy.ndarray`
            Array of bin indices.
        """
        coord = np.array(coord)[..., np.newaxis]
        is_equal = coord == self._labels

        if not np.all(np.any(is_equal, axis=-1)):
            label = coord[~np.any(is_equal, axis=-1)]
            raise ValueError(f"Not a valid label: {label}")

        return np.argmax(is_equal, axis=-1)

    def coord_to_pix(self, coord):
        """Transform label coordinate to pixel coordinate.

        Parameters
        ----------
        coord : `~numpy.ndarray`
            Array of axis label values.

        Returns
        -------
        pix : `~numpy.ndarray`
            Array of pixel coordinate values.
        """
        return self.coord_to_idx(coord).astype("float")

    def pix_to_idx(self, pix, clip=False):
        """Convert pixel to idx

        Parameters
        ----------
        pix : tuple of `~numpy.ndarray`
            Pixel coordinates.
        clip : bool, optional
            Choose whether to clip indices to the valid range of the
            axis. Default is False. If False, then indices for coordinates outside
            the axis range will be set to -1.

        Returns
        -------
        idx : tuple `~numpy.ndarray`
            Pixel indices.
        """
        if clip:
            idx = np.clip(pix, 0, self.nbin - 1)
        else:
            condition = (pix < 0) | (pix >= self.nbin)
            idx = np.where(condition, -1, pix)

        return idx

    @property
    def center(self):
        """Center of the label axis."""
        return self._labels

    @property
    def edges(self):
        """Edges of the label axis."""
        # TODO: Why not return self._labels here?
        raise ValueError("A LabelMapAxis does not define edges")

    @property
    def edges_min(self):
        """Edges of the label axis."""
        return self._labels

    @property
    def edges_max(self):
        """Edges of the label axis."""
        return self._labels

    @property
    def bin_width(self):
        """Bin width is unity."""
        return np.ones(self.nbin)

    @property
    def as_plot_xerr(self):
        """Return labels for plotting."""
        return 0.5 * np.ones(self.nbin)

    @property
    def as_plot_labels(self):
        """Return labels for plotting."""
        return self._labels.tolist()

    @property
    def as_plot_center(self):
        """Return labels for plotting."""
        return np.arange(self.nbin)

    @property
    def as_plot_edges(self):
        """Return labels for plotting."""
        return np.arange(self.nbin + 1) - 0.5

    def format_plot_xaxis(self, ax):
        """Format plot axis.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axis`
            Plot axis to format.

        Returns
        -------
        ax : `~matplotlib.pyplot.Axis`
            Formatted plot axis.
        """
        ax.set_xticks(self.as_plot_center)
        ax.set_xticklabels(
            self.as_plot_labels,
            rotation=30,
            ha="right",
            rotation_mode="anchor",
        )
        return ax

    def to_header(self, format="gadf", idx=0):
        """Create FITS header.

        Parameters
        ----------
        format : {"ogip"}
            Format specification. Default is "gadf".
        idx : int, optional
            Column index of the axis. Default is 0.

        Returns
        -------
        header : `~astropy.io.fits.Header`
            Header to extend.
        """
        header = fits.Header()

        if format == "gadf":
            key = f"AXCOLS{idx}"
            header[key] = self.name.upper()
        else:
            raise ValueError(f"Unknown format {format}")

        return header

    # TODO: how configurable should that be? column names?
    @classmethod
    def from_table(cls, table, format="gadf", idx=0):
        """Create time map axis from table.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Bin table HDU.
        format : {"gadf"}
            Format to use.
        idx : int
            Axis index. Default is 0.

        Returns
        -------
        axis : `TimeMapAxis`
            Time map axis.
        """
        if format == "gadf":
            colname = table.meta.get("AXCOLS{}".format(idx + 1))
            column = table[colname]
            if not np.issubdtype(column.dtype, np.str_):
                raise TypeError(f"Not a valid dtype for label axis: '{column.dtype}'")
            labels = np.unique(column.data)
        else:
            raise ValueError(f"Not a supported format: {format}")

        return cls(labels=labels, name=colname.lower())

    def __repr__(self):
        str_ = self.__class__.__name__ + "\n"
        str_ += "-" * len(self.__class__.__name__) + "\n\n"
        fmt = "\t{:<10s} : {:<10s}\n"
        str_ += fmt.format("name", self.name)
        str_ += fmt.format("nbins", str(self.nbin))
        str_ += fmt.format("node type", self.node_type)
        str_ += fmt.format("labels", "{0}".format(list(self._labels)))
        return str_.expandtabs(tabsize=2)

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    def is_allclose(self, other, **kwargs):
        """Check if other map axis is all close.

        Parameters
        ----------
        other : `LabelMapAxis`
            Other map axis.

        Returns
        -------
        is_allclose : bool
            Whether other axis is allclose.
        """
        if not isinstance(other, self.__class__):
            return TypeError(f"Cannot compare {type(self)} and {type(other)}")

        name_equal = self.name.upper() == other.name.upper()
        labels_equal = np.all(self.center == other.center)
        return name_equal & labels_equal

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.is_allclose(other=other)

    def __ne__(self, other):
        return not self.__eq__(other)

    # TODO: could create sub-labels here using dashes like "label-1-a", etc.
    def upsample(self, *args, **kwargs):
        """Not supported for label axis."""
        raise NotImplementedError("Upsampling a LabelMapAxis is not supported")

    # TODO: could merge labels here like "label-1-label2", etc.
    def downsample(self, *args, **kwargs):
        """Not supported for label axis."""
        raise NotImplementedError("Downsampling a LabelMapAxis is not supported")

    # TODO: could merge labels here like "label-1-label2", etc.
    def resample(self, *args, **kwargs):
        """Not supported for label axis."""
        raise NotImplementedError("Resampling a LabelMapAxis is not supported")

    # TODO: could create new labels here like "label-10-a"
    def pad(self, *args, **kwargs):
        """Not supported for label axis."""
        raise NotImplementedError("Padding a LabelMapAxis is not supported")

    def copy(self):
        """Copy the axis."""
        return copy.deepcopy(self)

    def slice(self, idx):
        """Create a new axis object by extracting a slice from this axis.

        Parameters
        ----------
        idx : slice
            Slice object selecting a sub-selection of the axis.

        Returns
        -------
        axis : `~LabelMapAxis`
            Sliced axis object.
        """
        return self.__class__(
            labels=self._labels[idx],
            name=self.name,
        )

    @classmethod
    def from_stack(cls, axes):
        """Create a label map axis by merging a list of label axis.

        Parameters
        ----------
        axes : list of `LabelMapAxis`
            List of label map axis to be merged.

        Returns
        -------
        axis : `LabelMapAxis`
            Stacked axis.
        """

        axis_stacked = axes[0]

        for ax in axes[1:]:
            axis_stacked = axis_stacked.concatenate(ax)

        return axis_stacked

    def concatenate(self, axis):
        """Concatenate another label map axis to this one into a new instance of `LabelMapAxis`.

        Names must agree between the axes. Labels must be unique.

        Parameters
        ----------
        axis : `LabelMapAxis`
            Axis to concatenate with.

        Returns
        -------
        axis : `LabelMapAxis`
            Concatenation of the two axis.
        """
        if not isinstance(axis, LabelMapAxis):
            raise TypeError(
                f"axis must be an instance of LabelMapAxis, got {axis.__class__.__name__} instead."
            )

        if self.name != axis.name:
            raise ValueError(f"Names must agree, got {self.name} and {axis.name} ")

        merged_labels = np.append(self.center, axis.center)

        return LabelMapAxis(merged_labels, self.name)

    def squash(self):
        """Create a new axis object by squashing the axis into one bin.

        The label of the new axis is given as "first-label...last-label".

        Returns
        -------
        axis : `LabelMapAxis`
            Squashed label map axis.
        """
        return LabelMapAxis(
            labels=[self.center[0] + "..." + self.center[-1]], name=self._name
        )

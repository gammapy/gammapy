# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility functions and classes for n-dimensional data and axes.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import itertools
from collections import OrderedDict
import numpy as np
from astropy.units import Quantity
from .array import array_stats_str

__all__ = ["NDDataArray", "DataAxis", "BinnedDataAxis", "sqrt_space"]


class NDDataArray(object):
    """ND Data Array Base class

    for usage examples see :gp-extra-notebook:`nddata_demo`

    Parameters
    ----------
    axes : list
        List of `~gammapy.utils.nddata.DataAxis`
    data : `~astropy.units.Quantity`
        Data
    meta : dict
        Meta info
    interp_kwargs : dict
        TODO
    """

    default_interp_kwargs = dict(bounds_error=False)
    """Default interpolation kwargs used to initialize the
    `scipy.interpolate.RegularGridInterpolator`.  The interpolation behaviour
    of an individual axis ('log', 'linear') can be passed to the axis on
    initialization."""

    def __init__(self, axes, data=None, meta=None, interp_kwargs=None):
        self._axes = axes
        if data is not None:
            self.data = data
        if meta is not None:
            self.meta = OrderedDict(meta)
        self.interp_kwargs = interp_kwargs or self.default_interp_kwargs

        self._regular_grid_interp = None

    def __str__(self):
        ss = "NDDataArray summary info\n"
        for axis in self.axes:
            ss += array_stats_str(axis.nodes, axis.name)
        ss += array_stats_str(self.data, "Data")
        return ss

    @property
    def axes(self):
        """Array holding the axes in correct order"""
        return self._axes

    def axis(self, name):
        """Return axis by name"""
        try:
            idx = [_.name for _ in self.axes].index(name)
        except ValueError:
            raise ValueError("Axis {} not found".format(name))
        return self.axes[idx]

    @property
    def data(self):
        """Array holding the n-dimensional data."""
        return self._data

    @data.setter
    def data(self, data):
        """Set data.

        Some sanity checks are performed to avoid an invalid array.
        Also, the interpolator is set to None to avoid unwanted behaviour.

        Parameters
        ----------
        data : `~astropy.units.Quantity`, array-like
            Data array
        """
        data = Quantity(data)
        dimension = len(data.shape)
        if dimension != self.dim:
            raise ValueError(
                "Overall dimensions to not match. "
                "Data: {}, Hist: {}".format(dimension, self.dim)
            )

        for dim in np.arange(self.dim):
            axis = self.axes[dim]
            if axis.nbins != data.shape[dim]:
                msg = "Data shape does not match in dimension {d}\n"
                msg += "Axis {n} : {sa}, Data {sd}"
                raise ValueError(
                    msg.format(d=dim, n=axis.name, sa=axis.nbins, sd=data.shape[dim])
                )
        self._regular_grid_interp = None
        self._data = data

    @property
    def dim(self):
        """Dimension (number of axes)"""
        return len(self.axes)

    def find_node(self, **kwargs):
        """Find next node

        Parameters
        ----------
        kwargs : dict
            Keys are the axis names, Values the evaluation points
        """
        node = []
        for axis in self.axes:
            lookup_val = Quantity(kwargs.pop(axis.name))
            temp = axis.find_node(lookup_val)
            node.append(temp)
        return node

    def evaluate(self, method=None, **kwargs):
        """Evaluate NDData Array

        This function provides a uniform interface to several interpolators.
        The evaluation nodes are given as ``kwargs``.

        Currently available:
        `~scipy.interpolate.RegularGridInterpolator`, methods: linear, nearest

        Parameters
        ----------
        method : str {'linear', 'nearest'}, optional
            Interpolation method
        kwargs : dict
            Keys are the axis names, Values the evaluation points

        Returns
        -------
        array : `~astropy.units.Quantity`
            Interpolated values, axis order is the same as for the NDData array
        """
        values = []
        for axis in self.axes:
            # Extract values for each axis, default: nodes
            temp = Quantity(kwargs.pop(axis.name, axis.nodes))
            # Transform to correct unit
            temp = temp.to(axis.unit).value
            # Transform to match interpolation behaviour of axis
            values.append(np.atleast_1d(axis._interp_values(temp)))

        # This is to catch e.g. typos in axis names
        if kwargs != {}:
            raise ValueError("Input given for unknown axis: {}".format(kwargs))

        # This is necessary since np.append does not support the 1D case
        if self.dim > 1:
            shapes = np.concatenate([np.shape(_) for _ in values])
        else:
            shapes = values[0].shape

        # Flatten in order to support 2D array input
        values = [_.flatten() for _ in values]
        points = list(itertools.product(*values))

        if self._regular_grid_interp is None:
            self._add_regular_grid_interp()

        method = method or self.default_interp_kwargs.get("method", None)
        res = self._regular_grid_interp(points, method=method, **kwargs)

        out = np.reshape(res, shapes).squeeze()

        # Clip interpolated values to be non-negative
        np.clip(out, 0, None, out=out)
        # Attach units to the output
        out = out * self.data.unit

        return out

    def evaluate_at_coord(self, points, method="linear", **kwargs):
        """Evaluate NDData Array on set of points.

        TODO: merge with `evaluate`?
        This method was added to support evaluating on arbitrary arrays
        of coordinates, not just on the outer product like `evaluate`.

        Parameters
        ----------
        points: dict
            contains the coordinates on which you want to interpolate (axis_name: value)
        method : str {'linear', 'nearest'}, optional
            Interpolation method
        kwargs : dict
            Keys are the axis names, Values the evaluation points

        Returns
        -------
        array : `~astropy.units.Quantity`
            Interpolated values, axis order is the same as for the NDData array
        """
        if self._regular_grid_interp is None:
            self._add_regular_grid_interp()

        points = tuple(
            [
                axis._interp_values(points[axis.name].to(axis.unit).value)
                for axis in self.axes
            ]
        )
        res = self._regular_grid_interp(points, method=method, **kwargs)

        # Clip interpolated values to be non-negative
        np.clip(res, 0, None, out=res)
        # Attach units to the output
        res = res * self.data.unit

        return res

    def _add_regular_grid_interp(self, interp_kwargs=None):
        """Add `~scipy.interpolate.RegularGridInterpolator`

        http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.RegularGridInterpolator.html

        Parameters
        ----------
        interp_kwargs : dict, optional
            Interpolation kwargs
        """
        from scipy.interpolate import RegularGridInterpolator

        if interp_kwargs is None:
            interp_kwargs = self.interp_kwargs
        points = [a._interp_nodes() for a in self.axes]

        values = self.data.value

        # If values contains nan, only setup interpolator in valid range
        if np.isnan(values).any():
            if self.dim > 1:
                raise NotImplementedError(
                    "Data grid contains nan. This is not"
                    "supported for arrays dimension > 1"
                )
            else:
                mask = np.isfinite(values)
                points = [points[0][mask]]
                values = values[mask]

        self._regular_grid_interp = RegularGridInterpolator(
            points, values, **interp_kwargs
        )


class DataAxis(object):
    """Data axis to be used with NDDataArray

    Axis values are interpreted as nodes.

    For binned data see `~gammapy.utils.nddata.BinnedDataAxis`.

    Parameters
    ----------
    nodes : `~astropy.units.Quantity`
        Interpolation nodes
    name : str, optional
        Axis name, default: 'Default'
    interpolation_mode : str {'linear', 'log'}
        Interpolation behaviour, default: 'linear'
    """

    def __init__(self, nodes, name="Default", interpolation_mode="linear"):
        # Need this for subclassing (see BinnedDataAxis)
        if nodes is not None:
            self._data = Quantity(nodes)
        self.name = name
        self._interpolation_mode = interpolation_mode

    def __str__(self):
        ss = self.__class__.__name__
        ss += "\nName: {}".format(self.name)
        ss += "\nUnit: {}".format(self.unit)
        ss += "\nNodes: {}".format(self.nbins)
        ss += "\nInterpolation mode: {}".format(self.interpolation_mode)

        return ss

    @property
    def unit(self):
        """Axis unit"""
        return self.nodes.unit

    @classmethod
    def logspace(cls, vmin, vmax, nbins, unit=None, **kwargs):
        """Create axis with equally log-spaced nodes

        if no unit is given, it will be taken from vmax,
        log interpolation is enable by default.

        Parameters
        ----------
        vmin : `~astropy.units.Quantity`, float
            Lowest value
        vmax : `~astropy.units.Quantity`, float
            Highest value
        bins : int
            Number of bins
        unit : `~astropy.units.UnitBase`, str
            Unit
        """
        kwargs.setdefault("interpolation_mode", "log")

        if unit is not None:
            vmin = Quantity(vmin, unit)
            vmax = Quantity(vmax, unit)
        else:
            vmin = Quantity(vmin)
            vmax = Quantity(vmax)
            unit = vmax.unit
            vmin = vmin.to(unit)

        x_min, x_max = np.log10([vmin.value, vmax.value])
        vals = np.logspace(x_min, x_max, nbins)

        return cls(vals * unit, **kwargs)

    def find_node(self, val):
        """Find next node

        Parameters
        ----------
        val : `~astropy.units.Quantity`
            Lookup value
        """
        val = Quantity(val)

        if not val.unit.is_equivalent(self.unit):
            raise ValueError(
                "Units mismatch: val.unit = {!r}, self.unit = {!r}".format(
                    val.unit, self.unit
                )
            )

        val = val.to(self.nodes.unit)
        val = np.atleast_1d(val)
        x1 = np.array([val] * self.nbins).transpose()
        x2 = np.array([self.nodes] * len(val))
        temp = np.abs(x1 - x2)
        idx = np.argmin(temp, axis=1)
        return idx

    @property
    def nbins(self):
        """Number of bins"""
        return len(self.nodes)

    @property
    def nodes(self):
        """Evaluation nodes"""
        return self._data

    @property
    def interpolation_mode(self):
        """Interpolation mode
        """
        return self._interpolation_mode

    def _interp_nodes(self):
        """Nodes to be used for interpolation"""
        if self.interpolation_mode == "log":
            return np.log10(self.nodes.value)
        else:
            return self.nodes.value

    def _interp_values(self, values):
        """Transform values correctly for interpolation"""
        if self.interpolation_mode == "log":
            return np.log10(values)
        else:
            return values


class BinnedDataAxis(DataAxis):
    """Data axis for binned data

    Parameters
    ----------
    lo : `~astropy.units.Quantity`
        Lower bin edges
    hi : `~astropy.units.Quantity`
        Upper bin edges
    name : str, optional
        Axis name, default: 'Default'
    interpolation_mode : str {'linear', 'log'}
        Interpolation behaviour, default: 'linear'
    """

    def __init__(self, lo, hi, **kwargs):
        self.lo = Quantity(lo)
        self.hi = Quantity(hi)
        super(BinnedDataAxis, self).__init__(None, **kwargs)

    @classmethod
    def logspace(cls, emin, emax, nbins, unit=None, **kwargs):
        # TODO: splitout log space into a helper function
        vals = DataAxis.logspace(emin, emax, nbins + 1, unit)._data
        return cls(vals[:-1], vals[1:], **kwargs)

    def __str__(self):
        ss = super(BinnedDataAxis, self).__str__()
        ss += "\nLower bounds {}".format(self.lo)
        ss += "\nUpper bounds {}".format(self.hi)

        return ss

    @property
    def bins(self):
        """Bin edges"""
        unit = self.lo.unit
        val = np.append(self.lo.value, self.hi.to(unit).value[-1])
        return val * unit

    @property
    def bin_width(self):
        """Bin width"""
        return self.hi - self.lo

    @property
    def nodes(self):
        """Evaluation nodes.

        Depending on the interpolation mode, either log or lin center are
        returned
        """
        if self.interpolation_mode == "log":
            return self.log_center()
        else:
            return self.lin_center()

    def lin_center(self):
        """Linear bin centers"""
        return (self.lo + self.hi) / 2

    def log_center(self):
        """Logarithmic bin centers"""
        return np.sqrt(self.lo * self.hi)


def sqrt_space(start, stop, num):
    """Return numbers spaced evenly on a square root scale.

    This function is similar to `numpy.linspace` and `numpy.logspace`.

    Parameters
    ----------
    start : float
        start is the starting value of the sequence
    stop : float
        stop is the final value of the sequence
    num : int
        Number of samples to generate.

    Returns
    -------
    samples : `~numpy.ndarray`
        1D array with a square root scale

    Examples
    --------
    >>> from gammapy.utils.nddata import sqrt_space
    >>> samples = sqrt_space(0, 2, 5)
    array([ 0.        ,  1.        ,  1.41421356,  1.73205081,  2.        ])

    """
    samples2 = np.linspace(start ** 2, stop ** 2, num)
    samples = np.sqrt(samples2)
    return samples

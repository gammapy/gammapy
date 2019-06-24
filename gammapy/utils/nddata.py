# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility functions and classes for n-dimensional data and axes."""
from collections import OrderedDict
import numpy as np
from astropy.units import Quantity
from .array import array_stats_str
from .interpolation import ScaledRegularGridInterpolator

__all__ = ["NDDataArray", "sqrt_space"]


class NDDataArray:
    """ND Data Array Base class

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

    default_interp_kwargs = dict(bounds_error=False, values_scale="lin")
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
            ss += str(axis)
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
            if axis.nbin != data.shape[dim]:
                msg = "Data shape does not match in dimension {d}\n"
                msg += "Axis {n} : {sa}, Data {sd}"
                raise ValueError(
                    msg.format(d=dim, n=axis.name, sa=axis.nbin, sd=data.shape[dim])
                )
        self._regular_grid_interp = None
        self._data = data

    @property
    def dim(self):
        """Dimension (number of axes)"""
        return len(self.axes)

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
        for idx, axis in enumerate(self.axes):
            # Extract values for each axis, default: nodes
            shape = [1] * len(self.axes)
            shape[idx] = -1
            default = axis.center.reshape(tuple(shape))
            temp = Quantity(kwargs.pop(axis.name, default))
            values.append(np.atleast_1d(temp))

        # This is to catch e.g. typos in axis names
        if kwargs != {}:
            raise ValueError("Input given for unknown axis: {}".format(kwargs))

        if self._regular_grid_interp is None:
            self._add_regular_grid_interp()

        return self._regular_grid_interp(values, method=method, **kwargs)

    def _add_regular_grid_interp(self, interp_kwargs=None):
        """Add `~scipy.interpolate.RegularGridInterpolator`

        http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.RegularGridInterpolator.html

        Parameters
        ----------
        interp_kwargs : dict, optional
            Interpolation kwargs
        """
        if interp_kwargs is None:
            interp_kwargs = self.interp_kwargs

        points = [a.center for a in self.axes]
        points_scale = [a.interp for a in self.axes]
        self._regular_grid_interp = ScaledRegularGridInterpolator(
            points, self.data, points_scale=points_scale, **interp_kwargs
        )


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
    >>> sqrt_space(0, 2, 5)
    array([0.        , 1.        , 1.41421356, 1.73205081, 2.        ])
    """
    return np.sqrt(np.linspace(start ** 2, stop ** 2, num))

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility functions and classes for n-dimensional data and axes."""
import numpy as np
from astropy.units import Quantity
from .array import array_stats_str
from .interpolation import ScaledRegularGridInterpolator

__all__ = ["NDDataArray"]


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
        from gammapy.maps.geom import MapAxes

        self._axes = MapAxes(axes)

        if np.shape(data) != self._axes.shape:
            raise ValueError(
                f"data shape {data.shape} does not match"
                f"axes shape {self._axes.shape}"
            )

        if data is not None:
            self.data = data

        self.meta = meta or {}
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
        self._regular_grid_interp = None
        self._data = data

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
            raise ValueError(f"Input given for unknown axis: {kwargs}")

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

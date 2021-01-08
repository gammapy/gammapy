# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility functions and classes for n-dimensional data and axes."""
import numpy as np
from astropy import units as u
from astropy.utils import lazyproperty
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
        data = u.Quantity(data)
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
        coords = self.axes.get_coord()

        for key, value in coords.items():
            coord = u.Quantity(kwargs.get(key, value))
            coords[key] = coord

        return self._interpolate(coords.values(), method=method)

    @property
    def _interpolate(self):
        points = [a.center for a in self.axes]
        points_scale = [a.interp for a in self.axes]
        return ScaledRegularGridInterpolator(
            points, self.data, points_scale=points_scale, **self.interp_kwargs
        )

    @lazyproperty
    def _integrate_rad(self):
        rad_axis = self.axes["rad"]
        rad_drad = (
                2 * np.pi * rad_axis.center * self.data * rad_axis.bin_width
        )
        values = rad_drad.cumsum().to_value("")
        values = np.insert(values, 0, 0)

        return ScaledRegularGridInterpolator(
            points=(rad_axis.edges,), values=values, fill_value=1,
        )

    # TODO: make this a more general integration method
    def integral_rad(self, rad_min, rad_max):
        """Compute radial integral, requires a "rad" axis to be defined.

        Parameters
        ----------
        rad_min, rad_max : `~astropy.units.Quantity`
            Integration limits.

        Returns
        -------
        integral : `~astropy.units.Quantity`
            Integral
        """
        integral_min = self._integrate_rad((rad_min,))
        integral_max = self._integrate_rad((rad_max,))
        return integral_max - integral_min
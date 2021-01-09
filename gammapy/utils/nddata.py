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
    axes : list or `MapAxes`
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
            self._data = u.Quantity(data)

        self.meta = meta or {}
        self.interp_kwargs = interp_kwargs or self.default_interp_kwargs

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
        """Set data

        Parameters
        ----------
        data : `~astropy.units.Quantity`, array-like
            Data array
        """
        self._data = u.Quantity(data)
        del self.__dict__["_interpolate"]
        del self.__dict__["_integrate_rad"]

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
            coord = kwargs.get(key, value)
            if coord is not None:
                coords[key] = u.Quantity(coord)

        return self._interpolate(coords.values(), method=method)

    @lazyproperty
    def _interpolate(self):
        points = [a.center for a in self.axes]
        points_scale = [a.interp for a in self.axes]
        return ScaledRegularGridInterpolator(
            points, self.data, points_scale=points_scale, **self.interp_kwargs
        )

    # TODO: define a proper integration method
    @lazyproperty
    def _integrate_rad(self):
        rad_axis = self.axes["rad"]
        rad_drad = (
                2 * np.pi * rad_axis.center * self.data * rad_axis.bin_width
        )
        idx_rad = self.axes.index("rad")
        values = rad_drad.cumsum(axis=idx_rad).to_value("")
        values = np.insert(values, 0, 0, axis=idx_rad)

        points = [ax.center for ax in self.axes]
        points[idx_rad] = rad_axis.edges
        return ScaledRegularGridInterpolator(
            points=points, values=values, fill_value=1,
        )

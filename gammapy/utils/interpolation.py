# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Interpolation utilities"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from astropy import units as u


__all__ = ["ScaledRegularGridInterpolator", "interpolation_scale"]


class ScaledRegularGridInterpolator(object):
    """Thin wrapper around `scipy.interpolate.RegularGridInterpolator`.

    The values are scaled before the interpolation and back-scaled after the
    interpolation.

    Parameters
    ----------
    points : tuple
        Tuple of points passed to `RegularGridInterpolator`.
    values :
        Values passed to `RegularGridInterpolator`.
    values_scale : {'lin', 'log', 'sqrt'}
        Interpolation scaling applied to values. If the values vary over many magnitudes
        a 'log' scaling is recommended.
    **kwargs : dict
        Keyword arguments passed to `RegularGridInterpolator`.
    """

    # TODO: add points scaling or axis scaling argument
    def __init__(self, points, values, values_scale="lin", extrapolate=True, **kwargs):

        if isinstance(values, u.Quantity):
            self._values_unit = values.unit
            values = values.value
        else:
            self._values_unit = None

        self.scale = interpolation_scale(values_scale)
        values_scaled = self.scale(values)

        if extrapolate:
            kwargs.setdefault("bounds_error", False)
            kwargs.setdefault("fill_value", None)

        self._interpolate = RegularGridInterpolator(
            points=points, values=values_scaled, **kwargs
        )

    def __call__(self, points, method="linear", clip=True, **kwargs):
        """Interpolate data points.

        Parameters
        ----------
        points : tuple of `np.ndarray`
            Tuple of coordinate arrays of the form (x_1, x_2, x_3, ...). Arrays are
            broadcasted internally.
        method : {"linear", "nearest"}
            Linear or nearest neighbour interpolation.
        clip : bool
            Clip values at zero after interpolation.
        """
        points = np.broadcast_arrays(*points)
        points_interp = np.stack([_.flat for _ in points]).T

        values = self._interpolate(points_interp, method, **kwargs)
        values = self.scale.inverse(values.reshape(points[0].shape))

        if clip:
            values = np.clip(values, 0, np.inf)

        if self._values_unit:
            return u.Quantity(values, self._values_unit, copy=False)
        else:
            return values


def interpolation_scale(scale="lin"):
    """Interpolation scaling.

    Parameters
    ----------
    scale : {"lin", "log", "sqrt"}
        Choose interpolation scaling.
    """
    if scale == "lin":
        return LinearScale()
    elif scale == "log":
        return LogScale()
    elif scale == "sqrt":
        return SqrtScale()
    else:
        raise ValueError("Not a valid value scaling mode.")


class LogScale(object):
    """Logarithmic scaling"""

    tiny = np.finfo(np.float32).tiny

    def __call__(self, values):
        values = np.clip(values, self.tiny, np.inf)
        return np.log(values)

    def inverse(self, values):
        return np.exp(values)


class SqrtScale(object):
    """Sqrt scaling"""

    def __call__(self, values):
        sign = np.sign(values)
        return sign * np.sqrt(sign * values)

    def inverse(self, values):
        return np.power(values, 2)


class LinearScale(object):
    """Linear scaling"""

    def __call__(self, values):
        return values

    def inverse(self, values):
        return values

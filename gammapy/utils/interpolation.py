# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Interpolation utilities"""
import numpy as np
import scipy.interpolate
from astropy import units as u

__all__ = [
    "ScaledRegularGridInterpolator",
    "interpolation_scale",
    "interpolate_profile",
]


class ScaledRegularGridInterpolator:
    """Thin wrapper around `scipy.interpolate.RegularGridInterpolator`.

    The values are scaled before the interpolation and back-scaled after the
    interpolation.

    Dimensions of length 1 are ignored in the interpolation of the data.

    Parameters
    ----------
    points : tuple of `~numpy.ndarray` or `~astropy.units.Quantity`
        Tuple of points passed to `RegularGridInterpolator`.
    values : `~numpy.ndarray`
        Values passed to `RegularGridInterpolator`.
    points_scale : tuple of str
        Interpolation scale used for the points.
    values_scale : {'lin', 'log', 'sqrt'}
        Interpolation scaling applied to values. If the values vary over many magnitudes
        a 'log' scaling is recommended.
    axis : int or None
        Axis along which to interpolate.
    **kwargs : dict
        Keyword arguments passed to `RegularGridInterpolator`.
    """

    def __init__(
        self,
        points,
        values,
        points_scale=None,
        values_scale="lin",
        extrapolate=True,
        axis=None,
        **kwargs,
    ):

        if points_scale is None:
            points_scale = ["lin"] * len(points)

        self.scale_points = [interpolation_scale(scale) for scale in points_scale]
        self.scale = interpolation_scale(values_scale)
        self._include_dim = [len(p) > 1 for p in points]

        points_scaled = tuple(
            [
                scale(p)
                for p, scale, _ in zip(points, self.scale_points, self._include_dim)
                if _
            ]
        )
        values_scaled = self.scale(values).squeeze()
        self.axis = axis

        if extrapolate:
            kwargs.setdefault("bounds_error", False)
            kwargs.setdefault("fill_value", None)

        if axis is None:
            # print(points_scaled, values_scaled)
            self._interpolate = scipy.interpolate.RegularGridInterpolator(
                points=points_scaled, values=values_scaled, **kwargs
            )
        else:
            self._interpolate = scipy.interpolate.interp1d(
                points_scaled[0], values_scaled, axis=axis
            )

    def __call__(self, points, method="linear", clip=True, **kwargs):
        """Interpolate data points.

        Parameters
        ----------
        points : tuple of `~numpy.ndarray` or `~astropy.units.Quantity`
            Tuple of coordinate arrays of the form (x_1, x_2, x_3, ...). Arrays are
            broadcasted internally.
        method : {"linear", "nearest"}
            Linear or nearest neighbour interpolation.
        clip : bool
            Clip values at zero after interpolation.
        """

        points = tuple(
            [
                scale(p)
                for scale, p, _ in zip(self.scale_points, points, self._include_dim)
                if _
            ]
        )

        if self.axis is None:
            points = np.broadcast_arrays(*points)
            points_interp = np.stack([_.flat for _ in points]).T
            values = self._interpolate(points_interp, method, **kwargs)
            values = self.scale.inverse(values.reshape(points[0].shape))
        else:
            values = self._interpolate(points[0])
            values = self.scale.inverse(values)

        if clip:
            values = np.clip(values, 0, np.inf)

        return values


def interpolation_scale(scale="lin"):
    """Interpolation scaling.

    Parameters
    ----------
    scale : {"lin", "log", "sqrt"}
        Choose interpolation scaling.
    """
    if scale in ["lin", "linear"]:
        return LinearScale()
    elif scale == "log":
        return LogScale()
    elif scale == "sqrt":
        return SqrtScale()
    else:
        raise ValueError(f"Not a valid value scaling mode: '{scale}'.")


class InterpolationScale:
    """Interpolation scale base class."""

    def __call__(self, values):
        if hasattr(self, "_unit"):
            values = values.to_value(self._unit)
        else:
            if isinstance(values, u.Quantity):
                self._unit = values.unit
                values = values.value
        return self._scale(values)

    def inverse(self, values):
        values = self._inverse(self, values)
        if hasattr(self, "_unit"):
            return u.Quantity(values, self._unit, copy=False)
        else:
            return values


class LogScale(InterpolationScale):
    """Logarithmic scaling"""

    tiny = np.finfo(np.float32).tiny

    def _scale(self, values):
        values = np.clip(values, self.tiny, np.inf)
        return np.log(values)

    @staticmethod
    def _inverse(self, values):
        output = np.exp(values)
        return np.where(abs(output) - self.tiny <= self.tiny, 0, output)


class SqrtScale(InterpolationScale):
    """Sqrt scaling"""

    @staticmethod
    def _scale(values):
        sign = np.sign(values)
        return sign * np.sqrt(sign * values)

    @staticmethod
    def _inverse(self, values):
        return np.power(values, 2)


class LinearScale(InterpolationScale):
    """Linear scaling"""

    @staticmethod
    def _scale(values):
        return values

    @staticmethod
    def _inverse(self, values):
        return values


def interpolate_profile(x, y, interp_scale="sqrt"):
    """Helper function to interpolate one-dimensional profiles.

    Parameters
    ----------
    x : `~numpy.ndarray`
        Array of x values
    y : `~numpy.ndarray`
        Array of y values
    interp_scale : {"sqrt", "lin"}
        Interpolation scale applied to the profile. If the profile is
        of parabolic shape, a "sqrt" scaling is recommended. In other cases or
        for fine sampled profiles a "lin" can also be used.

    Returns
    -------
    interp : `ScaledRegularGridInterpolator`
        Interpolator
    """
    sign = np.sign(np.gradient(y))
    return ScaledRegularGridInterpolator(
        points=(x,), values=sign * y, values_scale=interp_scale
    )

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Interpolation utilities"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


class ScaledRegularGridInterpolator(object):
    """Thin wrapper around `scipy.interpolate.RegularGridInterpolator`.

    Parameters
    ----------
    points : 

    values :

    values_scale : {'log', 'lin', 'sqrt'}
        Interpolation scaling applied to values. If the values vary over many magnitudes
        a 'log' scaling is recommended.
    **kwargs : dict
        Keyword arguments passed to `RegularGridInterpolator`.
    """

    def __init__(self, points, values, values_scale="lin", **kwargs):
        from scipy.interpolate import RegularGridInterpolator
        
        self.scale = interpolation_scale(values_scale)
        values_scaled = self.scale(values)

        kwargs.setdefault("fill_value", self.scale.fill_value)

        self._interpolate = RegularGridInterpolator(
                                    points=points,
                                    values=values_scaled,
                                    **kwargs
                                    )

    def __call__(self, points, method="linear", **kwargs):
        values = self._interpolate(points, method, **kwargs)
        return self.scale.inverse(values)


def interpolation_scale(scale="lin"):
    """Interpolation scaling.
    
    Paramaters
    ----------
    scale : {"lin", "log", "sqrt"}
        Choose interpolation scaling.
    """
    if scale == "lin":
        return LinearScaling()                
    elif scale == "log":
        return LogScaling()                
    elif scale == "sqrt":
        return SqrtScaling()                
    else:
        raise ValueError("Not a valid value scaling mode.")


class LogScaling(object):
    """Logarithmic scaling"""
    fill_value = -np.inf
    def __call__(self, values, clip=True):
        if clip:
            tiny = np.finfo(values.dtype).tiny
            values = values.copy()
            values[values <= 0] = tiny

        with np.errstate(divide="ignore"):
            return np.log(values)
    
    def inverse(self, values, clip=True):
        return np.exp(values)


class SqrtScaling(object):
    """Sqrt scaling"""
    fill_value = 0
    def __call__(self, values, clip=True):
        if clip:
            values = np.clip(values, 0, np.inf)
        return np.sqrt(values)
    
    def inverse(self, values, clip=True):
        values = np.power(values, 2)
        if clip:
            values = np.clip(values, 0, np.inf)
        return values


class LinearScaling(object):
    """Linear scaling"""
    fill_value = 0
    def __call__(self, values, clip=True):
        if clip:
            values = np.clip(values, 0, np.inf)
        return values
    
    def inverse(self, values, clip=True):
        if clip:
            values = np.clip(values, 0, np.inf)
        return values

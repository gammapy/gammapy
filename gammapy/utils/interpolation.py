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
    clip : bool
        Clip values at zero before interpolation.
    **kwargs : dict
        Keyword arguments passed to `RegularGridInterpolator`.
    """
    # TODO: add points scaling or axis scaling argument 
    def __init__(self, points, values, values_scale="lin", extrapolate=True, **kwargs):
        from scipy.interpolate import RegularGridInterpolator

        self.scale = interpolation_scale(values_scale)
        values_scaled = self.scale(values)

        if extrapolate:
            kwargs.setdefault("bounds_error", False)
            kwargs.setdefault("fill_value", None)
        
        self._interpolate = RegularGridInterpolator(
                                    points=points,
                                    values=values_scaled,
                                    **kwargs
                                    )

    def __call__(self, points, method="linear", clip=True, **kwargs):
        # the regular grid interpolator does not work with scalars, so we 
        # apply np.atleast_1d()
        if np.isscalar(points):
            values = self._interpolate([points], method, **kwargs)
        else:    
            values = self._interpolate(points, method, **kwargs)
        values = self.scale.inverse(values)
        
        if clip:
            np.clip(values, 0, np.inf, out=values)

        return values


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
    def __call__(self, values):
        tiny = np.finfo(values.dtype).tiny
        values = np.clip(values, tiny, np.inf)
        return np.log(values)
    
    def inverse(self, values):
        return np.exp(values)


class SqrtScaling(object):
    """Sqrt scaling"""
    def __call__(self, values):
        values = np.clip(values, 0, np.inf)
        return np.sqrt(values)
    
    def inverse(self, values):
        return np.power(values, 2)


class LinearScaling(object):
    """Linear scaling"""
    def __call__(self, values):
        return values
    
    def inverse(self, values):
        return values

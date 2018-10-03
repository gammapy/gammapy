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
        
        self.values_scale = values_scale

        #kwargs.setdefault("bounds_error", False)
        kwargs.setdefault("method", "linear")
        
        if values_scale == "log":
            values = values.copy()
            tiny = np.finfo(values.dtype).tiny
            values[values == 0] = tiny
            scale, scale_inv = np.log, np.exp
            kwargs.setdefault("fill_value", -np.inf)
        elif values_scale == "lin":
            scale, scale_inv = lambda x: x, lambda x: x
            kwargs.setdefault("fill_value", 0)
        elif values_scale == "sqrt":
            kwargs.setdefault("fill_value", 0)
            scale, scale_inv = np.sqrt, lambda x: x ** 2
        else:
            raise ValueError("Not a valid value scaling mode.")
        
        self._scale, self._scale_inv = scale, scale_inv

        if (values < 0).any() and values_scale != "lin":
            raise ValueError("Can't apply scaled interpolation to negative values."
                             "Choose 'lin' scaling.")

        values_scaled = self._scale(values)
        self._interpolate = RegularGridInterpolator(
                                    points=points,
                                    values=values_scaled,
                                    **kwargs
                                    )

    def __call__(self, points, method='linear', **kwargs):
        values = self._interpolate(points, method, **kwargs)
        return self._scale_inv(values)
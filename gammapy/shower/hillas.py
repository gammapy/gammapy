# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Hillas shower parametrization.
"""
from __future__ import print_function, division
import numpy as np

__all__ = ['hillas_parameters']


def hillas_parameters(x, y, s):
    """Compute Hillas parameters for a given shower image.

    Reference: Appendix of the Whipple Crab paper Weekes et al. (1998) 
    http://adsabs.harvard.edu/abs/1989ApJ...342..379W
    (corrected for some obvious typos) 

    Parameters
    ----------
    x : array_like
        Pixel x-coordinate
    y : array_like
        Pixel y-coordinate
    s : array_like
        Pixel value

    Returns
    -------
    hillas_parameters : dict
        Dictionary of Hillas parameters
    """
    x = np.asanyarray(x, dtype=np.float64)
    y = np.asanyarray(y, dtype=np.float64)
    s = np.asanyarray(s, dtype=np.float64)
    assert x.shape == s.shape
    assert y.shape == s.shape

    # Compute image moments
    _s = np.sum(s)
    m_x = np.sum(s * x) / _s
    m_y = np.sum(s * y) / _s
    m_xx = np.sum(s * x * x) / _s  # note: typo in paper
    m_yy = np.sum(s * y * y) / _s
    m_xy = np.sum(s * x * y) / _s  # note: typo in paper

    # Compute major axis line representation y = a * x + b
    S_xx = m_xx - m_x * m_x
    S_yy = m_yy - m_y * m_y
    S_xy = m_xy - m_x * m_y
    d = S_yy - S_xx
    temp = d * d + 4 * S_xy * S_xy
    a = (d + np.sqrt(temp)) / (2 * S_xy)
    b = m_y - a * m_x

    # Compute Hillas parameters
    width_2 = (S_yy + a * a * S_xx - 2 * a * S_xy) / (1 + a * a)
    width = np.sqrt(width_2)
    length_2 = (S_xx + a * a * S_yy + 2 * a * S_xy) / (1 + a * a)
    length = np.sqrt(length_2)
    miss = np.abs(b / (1 + a * a))
    r = np.sqrt(m_x * m_x + m_y * m_y)

    # Compute azwidth by transforming to (p, q) coordinates
    sin_theta = m_y / r
    cos_theta = m_x / r
    q = (m_x - x) * sin_theta + (y - m_y) * cos_theta
    m_q = np.sum(s * q) / _s
    m_qq = np.sum(s * q * q) / _s
    azwidth_2 = m_qq - m_q * m_q
    azwidth = np.sqrt(azwidth_2)

    # Return relevant parameters in a dict
    p = dict()
    p['x'] = m_x
    p['y'] = m_y
    p['a'] = a
    p['b'] = b
    p['width'] = width
    p['length'] = length
    p['miss'] = miss
    p['r'] = r
    p['azwidth'] = azwidth
    return p

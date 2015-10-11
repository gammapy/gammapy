# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Define the Gammapy matplotlib plotting style."""
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.visualization import astropy_mpl_style

__all__ = ['gammapy_mpl_style']

gammapy_mpl_style = astropy_mpl_style
"""
Gammapy matplotlib plotting style.
"""

gammapy_mpl_style['axes.color_cycle'] = [
    '#E24A33',  # orange
    '#348ABD',  # blue
    '#467821',  # green
    '#A60628',  # red
    '#7A68A6',  # purple
    '#CF4457',  # pink
    '#188487',  # turquoise
]
gammapy_mpl_style['interactive'] = False
gammapy_mpl_style['axes.labelcolor'] = '#565656'
gammapy_mpl_style['image.cmap'] = 'afmhot'
gammapy_mpl_style['axes.grid'] = False

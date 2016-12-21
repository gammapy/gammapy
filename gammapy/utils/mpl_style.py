# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Define the Gammapy matplotlib plotting style."""
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.visualization import astropy_mpl_style
from astropy.utils import minversion

# This returns False if matplotlib cannot be imported
MATPLOTLIB_GE_1_5 = minversion('matplotlib', '1.5')


__all__ = ['gammapy_mpl_style']

gammapy_mpl_style = astropy_mpl_style
"""
Gammapy matplotlib plotting style.
"""

color_cycle = ['#E24A33',  # orange
               '#348ABD',  # blue
               '#467821',  # green
               '#A60628',  # red
               '#7A68A6',  # purple
               '#CF4457',  # pink
               '#188487']  # turquoise

if MATPLOTLIB_GE_1_5:
    # This is a dependency of matplotlib, so should be present.
    from cycler import cycler
    gammapy_mpl_style['axes.prop_cycle'] = cycler('color', color_cycle)

else:
    gammapy_mpl_style['axes.color_cycle'] = color_cycle

gammapy_mpl_style['interactive'] = False
gammapy_mpl_style['axes.labelcolor'] = '#565656'
gammapy_mpl_style['image.cmap'] = 'afmhot'
gammapy_mpl_style['axes.grid'] = False

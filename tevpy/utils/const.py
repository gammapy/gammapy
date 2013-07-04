# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Thin wrapper and some additions to the astropy const and units packages."""

import numpy as np
from astropy.units import Unit

def conversion_factor(old, new):
    """Conversion factor from old to new units specified as strings.
    Example:
    >>> year_to_sec = cf('year', 'sec')
    """
    return Unit(old).to(Unit(new))

""" Astronomical constants """
d_sun_to_galactic_center = 8.5  # Distance Sun to Galactic center (kpc)

""" Additional constants """
sigma_to_fwhm = np.sqrt(8 * np.log(2))  # ~ 2.35
fwhm_to_sigma = 1 / sigma_to_fwhm

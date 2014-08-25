# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Thin wrapper and some additions to the astropy const and units packages."""
from __future__ import print_function, division
import numpy as np
from astropy.units import Unit, Quantity

__all__ = ['conversion_factor',
           'd_sun_to_galactic_center',
           'sigma_to_fwhm',
           'fwhm_to_sigma',
           ]


def conversion_factor(old, new):
    """Conversion factor from old to new units specified as strings.

    Examples
    --------
    >>> conversion_factor('year', 'second')
    31557600.0
    """
    return Unit(old).to(Unit(new))

""" Astronomical constants """
d_sun_to_galactic_center = Quantity(8.5, 'kpc')  # Distance Sun to Galactic center (kpc)

""" Additional constants """
sigma_to_fwhm = np.sqrt(8 * np.log(2))  # ~ 2.35
fwhm_to_sigma = 1 / sigma_to_fwhm

# Here's how to compute the conversion factors in astropy that I used in my old code:
# See https://gist.github.com/cdeil/5990465
# how they were computed and what units they are in
# TODO: use astropy constants and units throughout
#__all__ += ['c', 'h', 'h_eV', 'hbar', 'k_B', 'k_B_eV', 'm_H', 'm_e', 'm_e_eV',
#            'm_sun', 'sigma_T']

# from astropy import constants as const
c = 29979245800.0  # const.c.cgs.value
h = 6.62606896e-27  # const.h.cgs.value
h_eV = 4.13566722264e-15  # const.h.to('eV s').value
hbar = 1.054571628e-27  # const.hbar.cgs.value
k_B = 1.3806504e-16  # const.k_B.cgs.value
k_B_eV = 8.61734255962e-05  # const.k_B.to('eV/K').value
m_H = 1.660538782e-24  # const.m_p.cgs.value
m_e = 9.10938215e-28  # const.m_e.cgs.value
m_e_eV = 510998.895983  # (const.m_e * const.c ** 2).to('eV').value
m_sun = 1.9891e+33  # const.M_sun.cgs.value

sigma_T = 6.652458558e-25  # Thomson cross section

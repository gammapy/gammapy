# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Diffuse emission spectra.
"""
from __future__ import print_function, division
from astropy.units import Quantity

__all__ = ['diffuse_gamma_ray_flux']


def _power_law(E, N, k):
    E = Quantity(E, 'TeV')
    E0 = Quantity(1, 'TeV')
    N = Quantity(N, 'm^-2 s^-1 TeV^-1 sr^-1')
    flux = N * (E / E0) ** (-k)
    return flux


def diffuse_gamma_ray_flux(energy, component='isotropic'):
    """Diffuse gamma ray flux.

    TODO: describe available spectra.

    References:
    * 'isotropic':  http://adsabs.harvard.edu/abs/2010PhRvL.104j1101A

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Gamma-ray energy
    component : {'isotropic', 'bubble', 'galactic_fermi2', 'galactic_fermi4'}
        Diffuse model component

    Returns
    -------
    flux : `~astropy.units.Quantity`
        Gamma-ray flux in unit ``m^-2 s^-1 TeV^-1 sr^-1``
    """
    #flux = Quantity(1, 'm^-2 s^-1 TeV^-1 sr^-1')
    if component == 'isotropic':
        # Reference: abstract from this Fermi paper:
        # http://adsabs.harvard.edu/abs/2010PhRvL.104j1101A
        integral_flux = Quantity(1.03e-5, 'cm^-2 s^-1 sr^-1')
        gamma = 2.41
        return _power_law(energy, 1, 2)
    elif component == 'bubble':
        raise NotImplementedError
    elif component == 'galactic_fermi2':
        raise NotImplementedError
    else:
        raise ValueError('Invalid argument for component: {0}'.format(component))

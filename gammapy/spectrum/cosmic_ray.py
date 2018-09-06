# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simple models for cosmic ray spectra at Earth.

For measurements, the "Database of Charged Cosmic Rays (CRDB)" is a great resource:
http://lpsc.in2p3.fr/cosmic-rays-db/
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.units import Quantity

__all__ = ["cosmic_ray_flux"]


def _power_law(E, N, k):
    E = Quantity(E, "TeV")
    E0 = Quantity(1, "TeV")
    N = Quantity(N, "m^-2 s^-1 TeV^-1 sr^-1")
    flux = N * (E / E0) ** (-k)
    return flux


def _log_normal(E, L, E_p, w):
    E = Quantity(E, "TeV")
    E_p = Quantity(E_p, "TeV")
    L = Quantity(L, "m^-2 s^-1 sr^-1")
    term1 = L / (E * w * np.sqrt(2 * np.pi))
    term2 = np.exp(-np.log(E / E_p) ** 2 / (2 * w ** 2))
    return term1 * term2


def _electron_spectrum(E, N, k, L, E_p, w):
    flux = _power_law(E, N, k)
    flux += _log_normal(E, L, E_p, w)
    return flux


def cosmic_ray_flux(energy, particle="proton"):
    """Cosmic ray flux at Earth.

    These are the spectra assumed in this CTA study:
    Table 3 in http://adsabs.harvard.edu/abs/2013APh....43..171B

    The hadronic spectra are simple power-laws, the electron spectrum
    is the sum of  a power law and a log-normal component to model the
    "Fermi shoulder".

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Particle energy
    particle : {'electron', 'proton', 'He', 'N', 'Si', 'Fe'}
        Particle type

    Returns
    -------
    flux : `~astropy.units.Quantity`
        Cosmic ray flux in unit ``m^-2 s^-1 TeV^-1 sr^-1``
    """
    pars = {
        "electron": {"N": 6.85e-5, "k": 3.21, "L": 3.19e-3, "E_p": 0.107, "w": 0.776},
        "proton": {"N": 0.096, "k": 2.70},
        "N": {"N": 0.0719, "k": 2.64},
        "Si": {"N": 0.0284, "k": 2.66},
        "Fe": {"N": 0.0134, "k": 2.63},
    }

    if particle == "electron":
        return _electron_spectrum(energy, **pars["electron"])
    elif particle in ["proton", "He", "N", "Si", "Fe"]:
        return _power_law(energy, **pars[particle])
    else:
        raise ValueError("Invalid argument for particle: {}".format(particle))

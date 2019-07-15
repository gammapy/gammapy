# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simple models for cosmic ray spectra at Earth.

For measurements, the "Database of Charged Cosmic Rays (CRDB)" is a great resource:
http://lpsc.in2p3.fr/cosmic-rays-db/
"""

from astropy import units as u
from .models import PowerLaw

__all__ = ["cosmic_ray_spectrum"]


def cosmic_ray_spectrum(particle="proton"):
    """Cosmic ray flux at Earth.

    These are the spectra assumed in this CTA study:
    Table 3 in https://ui.adsabs.harvard.edu/abs/2013APh....43..171B

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

    proton = {
        "amplitude": 0.096 * u.Unit("1 / (m2 s TeV sr)"),
        "index": 2.70,
        "reference": 1 * u.TeV,
    }

    N = {
        "amplitude": 0.0719 * u.Unit("1 / (m2 s TeV sr)"),
        "index": 2.64,
        "reference": 1 * u.TeV,
    }

    Si = {
        "amplitude": 0.0284 * u.Unit("1 / (m2 s TeV sr)"),
        "index": 2.66,
        "reference": 1 * u.TeV,
    }

    Fe = {
        "amplitude": 0.0134 * u.Unit("1 / (m2 s TeV sr)"),
        "index": 2.63,
        "reference": 1 * u.TeV,
    }

    electron = {
        "amplitude": 6.85e-5 * u.Unit("1 / (m2 s TeV sr)"),
        "index": 3.21,
        "reference": 1 * u.TeV,
        "L": 3.19e-3,
        "E_p": 0.107,
        "w": 0.776,
    }

    if particle == "proton":
        model = PowerLaw(**proton)
    elif particle == "N":
        model = PowerLaw(**N)
    elif particle == "Si":
        model = PowerLaw(**Si)
    elif particle == "Fe":
        model = PowerLaw(**Fe)
    #    elif particle == "electron":
    #        return _electron_spectrum(energy, **pars["electron"])
    else:
        raise ValueError("Invalid argument for particle: {}".format(particle))

    return model

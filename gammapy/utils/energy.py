# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy import units as u

__all__ = ["energy_logspace", "energy_logcenter"]


def energy_logspace(emin, emax, nbins, unit=None, per_decade=False):
    """Create energy with equal log-spacing (`~astropy.units.Quantity`).

    Parameters
    ----------
    emin, emax : `~astropy.units.Quantity`, float
        Energy range
    nbins : int
        Number of bins
    unit : `~astropy.units.UnitBase`, str
        Energy unit
    per_decade : bool
        Whether nbins is per decade.
    """
    if unit is not None:
        emin = u.Quantity(emin, unit)
        emax = u.Quantity(emax, unit)
    else:
        emin = u.Quantity(emin)
        emax = u.Quantity(emax)
        unit = emax.unit
        emin = emin.to(unit)

    x_min, x_max = np.log10([emin.value, emax.value])

    if per_decade:
        nbins = (x_max - x_min) * nbins

    energy = np.logspace(x_min, x_max, nbins)

    return u.Quantity(energy, unit, copy=False)


def energy_logcenter(e_edges):
    """Compute energy log center.

    Parameters
    ----------
    e_edges : `~astropy.units.Quantity`, float
        Energy edges.
    """
    return np.sqrt(e_edges[:-1] * e_edges[1:])

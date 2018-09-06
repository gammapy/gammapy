# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Units and Quantity related helper functions"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import astropy.units as u

__all__ = ["standardise_unit", "unit_from_fits_image_hdu"]

log = logging.getLogger("__name__")


def standardise_unit(unit):
    """Standardise unit.

    Changes applied by this function:

    * Drop "photon" == "ph" from the unit
    * Drop "count" == "ct" from the unit

    Parameters
    ----------
    unit : `~astropy.units.Unit` or str
        Any old unit

    Returns
    -------
    unit : `~astropy.units.Unit`
        Shiny new, standardised unit

    Examples
    --------
    >>> from gammapy.utils.units import standardise_unit
    >>> standardise_unit('ph cm-2 s-1')
    Unit("1 / (cm2 s)")
    >>> standardise_unit('ct cm-2 s-1')
    Unit("1 / (cm2 s)")
    >>> standardise_unit('cm-2 s-1')
    Unit("1 / (cm2 s)")
    """
    unit = u.Unit(unit)
    bases, powers = [], []
    for base, power in zip(unit.bases, unit.powers):
        if str(base) not in {"ph", "ct"}:
            bases.append(base)
            powers.append(power)

    return u.CompositeUnit(scale=unit.scale, bases=bases, powers=powers)


def unit_from_fits_image_hdu(header):
    """Read unit from a FITS image HDU.

    - The ``BUNIT`` key is used.
    - `astropy.units.Unit` is called.
      If the ``BUNIT`` value is invalid, a log warning
      is emitted and the empty unit is used.
    - `standardise_unit` is called
    """
    unit = header.get("BUNIT", "")

    try:
        u.Unit(unit)
    except ValueError:
        log.warning(
            "Invalid value BUNIT={!r} in FITS header. Setting empty unit.".format(unit)
        )
        unit = ""

    return standardise_unit(unit)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import astropy.units as u
from .core import PSF

__all__ = ["PSF3D"]

log = logging.getLogger(__name__)


class PSF3D(PSF):
    """PSF with axes: energy, offset, rad.

    Data format specification: :ref:`gadf:psf_table`

    Parameters
    ----------
    axes : list of `~gammapy.maps.MapAxis` or `~gammapy.maps.MapAxes`
        Required axes (in the given order) are:
            * energy_true (true energy axis)
            * migra (energy migration axis)
            * rad (rad axis)
    data : `~astropy.units.Quantity`
        PSF (3-dim with axes: psf[rad_index, offset_index, energy_index].
    unit : dict of str or `~astropy.units.Unit`
        Unit.
    is_pointlike : bool, optional
        Whether the IRF is point-like.
        True for point-like IRFs, False for full-containment.
        Default is False.
    fov_alignment : `~gammapy.irf.FoVAlignment`, optional
        The orientation of the field of view coordinate system.
        Default is FoVAlignment.RADEC.
    meta : dict
        Metadata dictionary.
    """

    tag = "psf_table"
    required_axes = ["energy_true", "offset", "rad"]
    default_unit = u.sr**-1

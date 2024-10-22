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
    energy_axis_true : `gammapy.maps.MapAxis`
        True energy axis.
    offset_axis : `gammapy.maps.MapAxis`
        Offset axis.
    rad_axis : `gammapy.maps.MapAxis`
        Rad axis.
    data : `astropy.units.Quantity`
        PSF (3-dim with axes: psf[rad_index, offset_index, energy_index].
    meta : dict
        Metadata dictionary.
    """

    tag = "psf_table"
    required_axes = ["energy_true", "offset", "rad"]
    default_unit = u.sr**-1

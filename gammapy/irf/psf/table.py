# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.table import Table
from gammapy.maps import MapAxis
from gammapy.utils.array import array_stats_str
from gammapy.utils.gauss import Gauss2DPDF
from gammapy.utils.scripts import make_path
from .core import PSF

__all__ = ["PSF3D"]

log = logging.getLogger(__name__)


class PSF3D(PSF):
    """PSF with axes: energy, offset, rad.

    Data format specification: :ref:`gadf:psf_table`

    Parameters
    ----------
    energy_axis_true : `MapAxis`
        True energy axis.
    offset_axis : `MapAxis`
        Offset axis
    rad_axis : `MapAxis`
        Rad axis
    data : `~astropy.units.Quantity`
        PSF (3-dim with axes: psf[rad_index, offset_index, energy_index]
    meta : dict
        Meta dict
    """
    tag = "psf_table"
    required_axes = ["energy_true", "offset", "rad"]

    def plot_psf_vs_rad(self, offset="0 deg", energy_true="1 TeV"):
        """Plot PSF vs rad.

        Parameters
        ----------
        energy_true : `~astropy.units.Quantity`
            Energy. Default energy = 1 TeV
        offset : `~astropy.coordinates.Angle`
            Offset in the field of view. Default offset = 0 deg
        """
        energy_true = np.atleast_1d(u.Quantity(energy_true))
        table = self.to_energy_dependent_table_psf(offset=offset)
        return table.plot_psf_vs_rad(energy=energy_true)

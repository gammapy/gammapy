# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.coordinates import Angle
from astropy import units as u
from astropy.units import Quantity
from gammapy.maps import MapAxes, MapAxis
from gammapy.utils.array import array_stats_str
from .table import EnergyDependentTablePSF
from .core import ParametricPSF

__all__ = ["PSFKing"]

log = logging.getLogger(__name__)


class PSFKing(ParametricPSF):
    """King profile analytical PSF depending on energy and offset.

    This PSF parametrisation and FITS data format is described here: :ref:`gadf:psf_king`.

    Parameters
    ----------
    energy_axis_true : `MapAxis`
        True energy axis
    offset_axis : `MapAxis`
        Offset axis
    gamma : `~numpy.ndarray`
        PSF parameter (2D)
    sigma : `~astropy.coordinates.Angle`
        PSF parameter (2D)
    meta : dict
        Meta data

    """

    tag = "psf_king"
    required_axes = ["energy_true", "offset"]
    par_names = ["gamma", "sigma"]
    par_units = ["", "deg"]

    @staticmethod
    def evaluate_direct(rad, gamma, sigma):
        """Evaluate the PSF model.

        Formula is given here: :ref:`gadf:psf_king`.

        Parameters
        ----------
        r : `~astropy.coordinates.Angle`
            Offset from PSF center used for evaluating the PSF on a grid
        gamma : `~astropy.units.Quantity`
            model parameter, no unit
        sigma : `~astropy.coordinates.Angle`
            model parameter

        Returns
        -------
        psf_value : `~astropy.units.Quantity`
            PSF value
        """
        sigma2 = sigma * sigma

        with np.errstate(divide="ignore"):
            term1 = 1 / (2 * np.pi * sigma2)
            term2 = 1 - 1 / gamma
            term3 = (1 + rad ** 2 / (2 * gamma * sigma2)) ** (-gamma)

        return term1 * term2 * term3

    def evaluate(self, energy=None, offset=None):
        """Evaluate analytic PSF parameters at a given energy and offset.

        Uses nearest-neighbor interpolation.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            energy value
        offset : `~astropy.coordinates.Angle`
            Offset in the field of view

        Returns
        -------
        values : `~astropy.units.Quantity`
            Interpolated value
        """
        param = dict()
        energy = Quantity(energy)
        offset = Angle(offset)

        # Find nearest energy value

        # Find nearest energy value
        i = np.argmin(np.abs(self.axes["energy_true"].center - energy))
        j = np.argmin(np.abs(self.axes["offset"].center - offset))

        # TODO: Use some kind of interpolation to get PSF
        # parameters for every energy and theta

        # Select correct gauss parameters for given energy and theta
        sigma = self.data["sigma"][i][j] * u.deg
        gamma = self.data["gamma"][i][j]

        param["sigma"] = sigma
        param["gamma"] = gamma
        return param

    def to_energy_dependent_table_psf(self, theta=None, rad=None):
        """Convert to energy-dependent table PSF.

        Parameters
        ----------
        theta : `~astropy.coordinates.Angle`
            Offset in the field of view. Default theta = 0 deg
        rad : `~astropy.coordinates.Angle`
            Offset from PSF center used for evaluating the PSF on a grid.
            Default offset = [0, 0.005, ..., 1.495, 1.5] deg.
        exposure : `~astropy.units.Quantity`
            Energy dependent exposure. Should be in units equivalent to 'cm^2 s'.
            Default exposure = 1.

        Returns
        -------
        table_psf : `~gammapy.irf.EnergyDependentTablePSF`
            Energy-dependent PSF
        """
        # self.energy is already the logcenter
        energy_axis_true = self.axes["energy_true"]

        # Defaults
        theta = theta if theta is not None else Angle(0, "deg")

        if rad is None:
            rad = Angle(np.arange(0, 1.5, 0.005), "deg")

        rad_axis = MapAxis.from_nodes(rad, name="rad")

        psf_value = Quantity(np.empty((energy_axis_true.nbin, len(rad))), "deg^-2")

        for idx, energy in enumerate(energy_axis_true.center):
            param_king = self.evaluate(energy, theta)
            val = self.evaluate_direct(rad, param_king["gamma"], param_king["sigma"])
            psf_value[idx] = Quantity(val, "deg^-2")

        return EnergyDependentTablePSF(
            axes=[energy_axis_true, rad_axis],
            data=psf_value.value,
            unit=psf_value.unit
        )

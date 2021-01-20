# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.coordinates import Angle
from astropy import units as u
from astropy.units import Quantity
from gammapy.maps import MapAxis
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
    required_parameters = ["gamma", "sigma"]

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

    def evaluate(self, energy_true, offset):
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
        energy = Quantity(energy_true)
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
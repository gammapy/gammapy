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
    default_interp_kwargs = dict(
        bounds_error=False, fill_value=None
    )

    def evaluate_parameters(self, energy_true, offset):
        """Evaluate analytic PSF parameters at a given energy and offset.

        Uses nearest-neighbor interpolation.

        Parameters
        ----------
        energy_true : `~astropy.units.Quantity`
            energy value
        offset : `~astropy.coordinates.Angle`
            Offset in the field of view

        Returns
        -------
        values : `~astropy.units.Quantity`
            Interpolated value
        """
        pars = {}
        for name in self.required_parameters:
            value = self._interpolators[name]((energy_true, offset))
            pars[name] = value

        return pars

    def evaluate(self, rad, energy_true, offset):
        """Evaluate the PSF model.

        Formula is given here: :ref:`gadf:psf_king`.

        Parameters
        ----------
        rad : `~astropy.coordinates.Angle`
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
        pars = self.evaluate_parameters(
            energy_true=energy_true, offset=offset
        )
        sigma, gamma = pars["sigma"], pars["gamma"]

        with np.errstate(divide="ignore"):
            term1 = 1 / (2 * np.pi * sigma ** 2)
            term2 = 1 - 1 / gamma
            term3 = (1 + rad ** 2 / (2 * gamma * sigma ** 2)) ** (-gamma)

        return term1 * term2 * term3


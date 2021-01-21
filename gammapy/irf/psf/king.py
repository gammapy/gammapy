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
    axes : list of `MapAxis` or `MapAxes`
        Data axes, required are ["energy_true", "offset"]
    meta : dict
        Meta data

    """

    tag = "psf_king"
    required_axes = ["energy_true", "offset"]
    required_parameters = ["gamma", "sigma"]
    default_interp_kwargs = dict(
        bounds_error=False, fill_value=None
    )

    def evaluate(self, rad, energy_true, offset):
        """Evaluate the PSF model.

        Formula is given here: :ref:`gadf:psf_king`.

        Parameters
        ----------
        rad : `~astropy.coordinates.Angle`
            Offset from PSF center used for evaluating the PSF on a grid

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


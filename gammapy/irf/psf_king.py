# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.table import Table
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.io import fits
from ..utils.scripts import make_path
from ..utils.array import array_stats_str
from ..utils.energy import Energy, EnergyBounds
from . import EnergyDependentTablePSF

__all__ = ["PSFKing"]

log = logging.getLogger(__name__)


class PSFKing(object):
    """King profile analytical PSF depending on energy and offset.

    This PSF parametrisation and FITS data format is described here: :ref:`gadf:psf_king`.

    Parameters
    ----------
    energy_lo : `~astropy.units.Quantity`
        Lower energy boundary of the energy bin.
    energy_hi : `~astropy.units.Quantity`
        Upper energy boundary of the energy bin.
    offset : `~astropy.coordinates.Angle`
        Offset nodes (1D)
    gamma : `~numpy.ndarray`
        PSF parameter (2D)
    sigma : `~astropy.coordinates.Angle`
        PSF parameter (2D)
    """

    def __init__(
        self,
        energy_lo,
        energy_hi,
        offset,
        gamma,
        sigma,
        energy_thresh_lo=Quantity(0.1, "TeV"),
        energy_thresh_hi=Quantity(100, "TeV"),
    ):
        self.energy_lo = energy_lo.to("TeV")
        self.energy_hi = energy_hi.to("TeV")
        self.offset = Angle(offset)
        ebounds = EnergyBounds.from_lower_and_upper_bounds(energy_lo, energy_hi)
        self.energy = ebounds.log_centers
        self.gamma = np.asanyarray(gamma)
        self.sigma = Angle(sigma)

        self.energy_thresh_lo = Quantity(energy_thresh_lo).to("TeV")
        self.energy_thresh_hi = Quantity(energy_thresh_hi).to("TeV")

    def info(self):
        """Print some basic info.
        """
        ss = "\nSummary PSFKing info\n"
        ss += "---------------------\n"
        ss += array_stats_str(self.offset, "offset")
        ss += array_stats_str(self.energy, "energy")
        ss += array_stats_str(self.gamma, "gamma")
        ss += array_stats_str(self.sigma, "sigma")

        # TODO: should quote containment values also

        return ss

    @classmethod
    def read(cls, filename, hdu=1):
        """Create `PSFKing` from FITS file.

        Parameters
        ----------
        filename : str
            File name
        """
        filename = str(make_path(filename))
        # TODO: implement it so that HDUCLASS is used
        # http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/hdu_index/index.html

        table = Table.read(filename, hdu=hdu)
        return cls.from_table(table)

        # hdu_list = fits.open(filename)
        # hdu = hdu_list[hdu]
        # return cls.from_fits(hdu)

    @classmethod
    def from_table(cls, table):
        """Create `PSFKing` from `~astropy.table.Table`.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table King PSF info.
        """
        offset_lo = table["THETA_LO"].quantity[0]
        offset_hi = table["THETA_HI"].quantity[0]
        offset = (offset_hi + offset_lo) / 2
        offset = Angle(offset, unit=table["THETA_LO"].unit)

        energy_lo = table["ENERG_LO"].quantity[0]
        energy_hi = table["ENERG_HI"].quantity[0]

        gamma = table["GAMMA"].quantity[0]
        sigma = table["SIGMA"].quantity[0]

        opts = {}
        try:
            opts["energy_thresh_lo"] = Quantity(table.meta["LO_THRES"], "TeV")
            opts["energy_thresh_hi"] = Quantity(table.meta["HI_THRES"], "TeV")
        except KeyError:
            pass

        return cls(energy_lo, energy_hi, offset, gamma, sigma, **opts)

    def to_fits(self):
        """
        Convert PSF table data to FITS HDU list.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            PSF in HDU list format.
        """
        # Set up data
        names = ["ENERG_LO", "ENERG_HI", "THETA_LO", "THETA_HI", "SIGMA", "GAMMA"]
        units = ["TeV", "TeV", "deg", "deg", "deg", ""]
        data = [
            self.energy_lo,
            self.energy_hi,
            self.offset,
            self.offset,
            self.sigma,
            self.gamma,
        ]

        table = Table()
        for name_, data_, unit_ in zip(names, data, units):
            table[name_] = [data_]
            table[name_].unit = unit_

        hdu = fits.BinTableHDU(table)
        hdu.header["LO_THRES"] = self.energy_thresh_lo.value
        hdu.header["HI_THRES"] = self.energy_thresh_hi.value

        return fits.HDUList([fits.PrimaryHDU(), hdu])

    def write(self, filename, *args, **kwargs):
        """Write PSF to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_fits().writeto(filename, *args, **kwargs)

    @staticmethod
    def evaluate_direct(r, gamma, sigma):
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
        r2 = r * r
        sigma2 = sigma * sigma

        with np.errstate(divide="ignore"):
            term1 = 1 / (2 * np.pi * sigma2)
            term2 = 1 - 1 / gamma
            term3 = (1 + r2 / (2 * gamma * sigma2)) ** (-gamma)

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
        energy = Energy(energy)
        offset = Angle(offset)

        # Find nearest energy value
        i = np.argmin(np.abs(self.energy - energy))
        j = np.argmin(np.abs(self.offset - offset))

        # TODO: Use some kind of interpolation to get PSF
        # parameters for every energy and theta

        # Select correct gauss parameters for given energy and theta
        sigma = self.sigma[j][i]
        gamma = self.gamma[j][i]

        param["sigma"] = sigma
        param["gamma"] = gamma
        return param

    def to_energy_dependent_table_psf(self, theta=None, rad=None, exposure=None):
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
        energies = self.energy

        # Defaults
        theta = theta if theta is not None else Angle(0, "deg")
        rad = rad if rad is not None else Angle(np.arange(0, 1.5, 0.005), "deg")
        psf_value = Quantity(np.empty((len(energies), len(rad))), "deg^-2")

        for i, energy in enumerate(energies):
            param_king = self.evaluate(energy, theta)
            val = self.evaluate_direct(rad, param_king["gamma"], param_king["sigma"])
            psf_value[i] = Quantity(val, "deg^-2")

        return EnergyDependentTablePSF(
            energy=energies, rad=rad, exposure=exposure, psf_value=psf_value
        )

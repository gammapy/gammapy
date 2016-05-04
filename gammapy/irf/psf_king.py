# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.table import Table
from astropy.units import Quantity
from astropy.coordinates import Angle
from ..utils.scripts import make_path
from ..utils.array import array_stats_str
from ..utils.energy import Energy
from . import EnergyDependentTablePSF

__all__ = ['PSFKing']


class PSFKing(object):
    """King profile analytical PSF depending on energy and theta.

    This PSF parametrisation and FITS data format is described here: :ref:`gadf:psf_king`.

    Parameters
    ----------
    offset : `~astropy.coordinates.Angle`
        Offset nodes (1D)
    energy : `~gammapy.utils.energy.Energy`
        Energy nodes (1D)
    gamma : `~numpy.ndarray`
        PSF parameter (2D)
    sigma : `~astropy.coordinates.Angle`
        PSF parameter (2D)
    """

    def __init__(self, offset, energy, gamma, sigma):
        self.offset = Angle(offset)
        self.energy = Energy(energy)
        self.gamma = np.asanyarray(gamma)
        self.sigma = Angle(sigma)

    def info(self):
        """Print some basic info.
        """
        ss = "\nSummary PSFKing info\n"
        ss += "---------------------\n"
        ss += array_stats_str(self.offset, 'offset')
        ss += array_stats_str(self.energy, 'energy')
        ss += array_stats_str(self.gamma, 'gamma')
        ss += array_stats_str(self.sigma, 'sigma')

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
        # http://gamma-astro-data-formats.readthedocs.org/en/latest/data_storage/hdu_index/index.html

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
        theta_lo = table['THETA_LO'].squeeze()
        theta_hi = table['THETA_HI'].squeeze()
        offset = (theta_hi + theta_lo) / 2
        offset = Angle(offset, unit=table['THETA_LO'].unit)

        energy_lo = table['ENERG_LO'].squeeze()
        energy_hi = table['ENERG_HI'].squeeze()
        energy = np.sqrt(energy_lo * energy_hi)
        energy = Energy(energy, unit=table['ENERG_LO'].unit)

        gamma = Quantity(table['GAMMA'].squeeze(), table['GAMMA'].unit)
        sigma = Quantity(table['SIGMA'].squeeze(), table['SIGMA'].unit)

        return cls(offset, energy, gamma, sigma)

    @staticmethod
    def evaluate_direct(r, gamma, sigma):
        """Evaluate formula from here: :ref:`gadf:psf_king`.

        Parameters
        ----------
        TODO

        Returns
        -------
        TODO
        """
        r2 = r * r
        sigma2 = sigma * sigma

        term1 = 1 / (2 * np.pi * sigma2)
        term2 = 1 - 1 / gamma
        term3 = (1 + r2 / 2 * gamma * sigma2) ** (-gamma)

        return term1 * term2 * term3

    def evaluate(self, energy=None, offset=None, interp_kwargs=None):
        """Interpolate the value of the `EnergyOffsetArray` at a given offset and Energy.

        Parameters
        ----------
        offset : `~astropy.coordinates.Angle`
            offset value
        energy : `~astropy.units.Quantity`
            energy value
        interp_kwargs : dict
            option for interpolation for `~scipy.interpolate.RegularGridInterpolator`

        Returns
        -------
        values : `~astropy.units.Quantity`
            Interpolated value
        """
        param=dict()
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

    def to_table_psf(self, theta=None, offset=None, exposure=None):
        """
        Convert triple Gaussian PSF ot table PSF.

        Parameters
        ----------
        theta : `~astropy.coordinates.Angle`
            Offset in the field of view. Default theta = 0 deg
        offset : `~astropy.coordinates.Angle`
            Offset from PSF center used for evaluating the PSF on a grid.
            Default offset = [0, 0.005, ..., 1.495, 1.5] deg.
        exposure : `~astropy.units.Quantity`
            Energy dependent exposure. Should be in units equivalent to 'cm^2 s'.
            Default exposure = 1.

        Returns
        -------
        tabe_psf : `~gammapy.irf.EnergyDependentTablePSF`
            Instance of `EnergyDependentTablePSF`.
        """
        # self.energy is already the logcenter
        energies = self.energy

        # Defaults
        theta = theta or Angle(0, 'deg')
        offset = offset or Angle(np.arange(0, 1.5, 0.005), 'deg')
        psf_value = Quantity(np.empty((len(energies), len(offset))), 'deg^-2')

        for i, energy in enumerate(energies):
            param_king = self.evaluate(energy, theta)
            psf_value[i] = Quantity(self.evaluate_direct(offset, param_king["gamma"], param_king["sigma"]), 'deg^-2')

        return EnergyDependentTablePSF(energy=energies, offset=offset,
                                       exposure=exposure, psf_value=psf_value)


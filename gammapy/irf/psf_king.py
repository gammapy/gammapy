# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.table import Table
from astropy.units import Quantity
from astropy.coordinates import Angle
from ..utils.energy import Energy
from ..utils.scripts import make_path
from ..utils.array import array_stats_str

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

    def evaluate(self, offset=None, energy=None, interp_kwargs=None):
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
        energy = Energy(energy)
        theta = Angle(theta)

        # Find nearest energy value
        i = np.argmin(np.abs(self.energy - energy))
        j = np.argmin(np.abs(self.offset - offset))

        # TODO: Use some kind of interpolation to get PSF
        # parameters for every energy and theta

        # Select correct gauss parameters for given energy and theta
        sigma = self.sigma[j][i]
        gamma = self.gamma[j][i]

        return self.evaluate_direct(r, gamma, sigma)

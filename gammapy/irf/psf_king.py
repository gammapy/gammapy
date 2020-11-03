# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity
from gammapy.maps import MapAxes, MapAxis
from gammapy.utils.array import array_stats_str
from gammapy.utils.scripts import make_path
from .psf_table import EnergyDependentTablePSF

__all__ = ["PSFKing"]

log = logging.getLogger(__name__)


class PSFKing:
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
    """

    tag = "psf_king"

    def __init__(
        self,
        energy_axis_true,
        offset_axis,
        gamma,
        sigma,
        energy_thresh_lo=Quantity(0.1, "TeV"),
        energy_thresh_hi=Quantity(100, "TeV"),
    ):
        assert energy_axis_true.name == "energy_true"
        self._energy_axis_true = energy_axis_true

        assert offset_axis.name == "offset"
        self._offset_axis = offset_axis

        self.gamma = np.asanyarray(gamma)
        self.sigma = Angle(sigma)

        self.energy_thresh_lo = Quantity(energy_thresh_lo).to("TeV")
        self.energy_thresh_hi = Quantity(energy_thresh_hi).to("TeV")

    @property
    def energy_axis_true(self):
        return self._energy_axis_true

    @property
    def offset_axis(self):
        return self._offset_axis

    def info(self):
        """Print some basic info.
        """
        ss = "\nSummary PSFKing info\n"
        ss += "---------------------\n"
        ss += array_stats_str(self.offset_axis.center, "offset")
        ss += array_stats_str(self.energy_axis_true.center, "energy")
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
        # TODO: implement it so that HDUCLASS is used
        # http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/hdu_index/index.html

        table = Table.read(make_path(filename), hdu=hdu)
        return cls.from_table(table)

    @classmethod
    def from_table(cls, table):
        """Create `PSFKing` from `~astropy.table.Table`.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table King PSF info.
        """
        energy_axis_true = MapAxis.from_table(
            table, column_prefix="ENERG", format="gadf-dl3"
        )
        offset_axis = MapAxis.from_table(
            table, column_prefix="THETA", format="gadf-dl3"
        )

        gamma = table["GAMMA"].quantity[0]
        sigma = table["SIGMA"].quantity[0]

        opts = {}
        try:
            opts["energy_thresh_lo"] = Quantity(table.meta["LO_THRES"], "TeV")
            opts["energy_thresh_hi"] = Quantity(table.meta["HI_THRES"], "TeV")
        except KeyError:
            pass

        return cls(
            energy_axis_true=energy_axis_true,
            offset_axis=offset_axis,
            gamma=gamma,
            sigma=sigma,
            **opts
        )

    def to_hdulist(self):
        """
        Convert PSF table data to FITS HDU list.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            PSF in HDU list format.
        """
        axes = MapAxes([self.energy_axis_true, self.offset_axis])
        table = axes.to_table(format="gadf-dl3")

        # Set up data
        names = ["SIGMA", "GAMMA"]
        units = ["deg", ""]
        data = [
            self.sigma,
            self.gamma,
        ]

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
        self.to_hdulist().writeto(str(make_path(filename)), *args, **kwargs)

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
        energy = Quantity(energy)
        offset = Angle(offset)

        # Find nearest energy value

        # Find nearest energy value
        i = np.argmin(np.abs(self.energy_axis_true.center - energy))
        j = np.argmin(np.abs(self.offset_axis.center - offset))

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
        energies = self.energy_axis_true.center

        # Defaults
        theta = theta if theta is not None else Angle(0, "deg")

        if rad is None:
            rad = Angle(np.arange(0, 1.5, 0.005), "deg")

        rad_axis = MapAxis.from_nodes(rad, name="rad")

        psf_value = Quantity(np.empty((len(energies), len(rad))), "deg^-2")

        for i, energy in enumerate(energies):
            param_king = self.evaluate(energy, theta)
            val = self.evaluate_direct(rad, param_king["gamma"], param_king["sigma"])
            psf_value[i] = Quantity(val, "deg^-2")

        return EnergyDependentTablePSF(
            energy_axis_true=self.energy_axis_true,
            rad_axis=rad_axis,
            exposure=exposure,
            psf_value=psf_value,
        )

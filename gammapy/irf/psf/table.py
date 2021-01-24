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

__all__ = ["EnergyDependentTablePSF", "PSF3D"]

log = logging.getLogger(__name__)


class EnergyDependentTablePSF(PSF):
    """Energy-dependent radially-symmetric table PSF (``gtpsf`` format).

    TODO: add references and explanations.

    Parameters
    ----------
    energy_axis_true : `MapAxis`
        Energy axis
    rad_axis : `MapAxis`
        Offset angle wrt source position axis
    exposure : `~astropy.units.Quantity`
        Exposure (1-dim)
    data : `~astropy.units.Quantity`
        PSF (2-dim with axes: psf[energy_index, offset_index]
    """
    required_axes = ["energy_true", "rad"]

    def __init__(
        self,
        axes,
        exposure=None,
        data=None,
        meta=None,
        unit=""
    ):
        super().__init__(axes=axes, data=data, meta=meta, unit=unit)

        if exposure is None:
            self.exposure = u.Quantity(np.ones(self.axes["energy_true"].nbin), "cm^2 s")
        else:
            self.exposure = u.Quantity(exposure).to("cm^2 s")

    @classmethod
    def from_hdulist(cls, hdu_list):
        """Create `EnergyDependentTablePSF` from ``gtpsf`` format HDU list.

        Parameters
        ----------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list with ``THETA`` and ``PSF`` extensions.
        """
        rad_axis = MapAxis.from_table_hdu(hdu_list["THETA"], format="gtpsf")

        table = Table.read(hdu_list["PSF"])
        energy_axis_true = MapAxis.from_table(table, format="gtpsf")
        exposure = table["Exposure"].data * u.Unit("cm2 s")

        data = table["Psf"].data
        return cls(
            axes=[energy_axis_true, rad_axis],
            exposure=exposure,
            data=data,
            unit="sr-1"
        )

    def to_hdulist(self):
        """Convert to FITS HDU list format.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            PSF in HDU list format.
        """
        rad_hdu = self.axes["rad"].to_table_hdu(format="gtpsf")
        psf_table = self.axes["energy_true"].to_table(format="gtpsf")

        psf_table["Exposure"] = self.exposure.to("cm^2 s")
        psf_table["Psf"] = self.quantity.to("sr^-1")
        psf_hdu = fits.BinTableHDU(data=psf_table, name="PSF")
        return fits.HDUList([fits.PrimaryHDU(), rad_hdu, psf_hdu])

    @classmethod
    def read(cls, filename):
        """Create `EnergyDependentTablePSF` from ``gtpsf``-format FITS file.

        Parameters
        ----------
        filename : str
            File name
        """
        with fits.open(str(make_path(filename)), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist)

    def write(self, filename, *args, **kwargs):
        """Write to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_hdulist().writeto(str(make_path(filename)), *args, **kwargs)

    def table_psf_in_energy_range(
        self, energy_range, spectrum=None, n_bins=11, **kwargs
    ):
        """Average PSF in a given energy band.

        Expected counts in sub energy bands given the given exposure
        and spectrum are used as weights.

        Parameters
        ----------
        energy_range : `~astropy.units.Quantity`
            Energy band
        spectrum : `~gammapy.modeling.models.SpectralModel`
            Spectral model used for weighting the PSF. Default is a power law
            with index=2.
        n_bins : int
            Number of energy points in the energy band, used to compute the
            weighted PSF.

        Returns
        -------
        psf : `EnergyDependentTablePSF`
            Table PSF
        """
        from gammapy.modeling.models import PowerLawSpectralModel, TemplateSpectralModel

        if spectrum is None:
            spectrum = PowerLawSpectralModel()

        exposure = TemplateSpectralModel(self.axes["energy_true"].center, self.exposure)

        e_min, e_max = energy_range
        energy = MapAxis.from_energy_bounds(e_min, e_max, n_bins).edges[:, np.newaxis]

        weights = spectrum(energy) * exposure(energy)
        weights /= weights.sum()

        psf_value = self.evaluate(energy_true=energy)
        psf_value_weighted = weights * psf_value

        energy_axis = MapAxis.from_edges(energy_range, name="energy_true")

        data = psf_value_weighted.sum(axis=0, keepdims=True)
        return self.__class__(
            axes=[energy_axis, self.axes["rad"]], data=data.value, unit=data.unit, **kwargs
        )

    def info(self):
        """Print basic info"""
        print(str(self))

    def plot_psf_vs_rad(self, energy=None, ax=None, **kwargs):
        """Plot PSF vs radius.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energies where to plot the PSF.
        **kwargs : dict
            Keyword arguments pass to `~matplotlib.pyplot.plot`.
        """
        import matplotlib.pyplot as plt

        if energy is None:
            energy = [100, 1000, 10000] * u.GeV

        ax = plt.gca() if ax is None else ax

        for value in energy:
            psf_value = np.squeeze(self.evaluate(energy_true=value))
            label = f"{value:.0f}"
            ax.plot(
                self.axes["rad"].center.to_value("deg"),
                psf_value.to_value("sr-1"),
                label=label,
                **kwargs,
            )

        ax.set_yscale("log")
        ax.set_xlabel("Offset (deg)")
        ax.set_ylabel("PSF (1 / sr)")
        plt.legend()
        return ax

    def plot_containment_vs_energy(
        self, ax=None, fractions=[0.68, 0.8, 0.95], **kwargs
    ):
        """Plot containment versus energy."""
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        energy_true = self.axes["energy_true"].center
        for fraction in fractions:
            rad = self.containment_radius(
                energy_true=energy_true, fraction=fraction
            )
            label = f"{100 * fraction:.1f}% Containment"
            ax.plot(
                energy_true.to_value("GeV"),
                rad.to_value("deg"),
                label=label,
                **kwargs,
            )

        ax.semilogx()
        ax.legend(loc="best")
        ax.set_xlabel("Energy (GeV)")
        ax.set_ylabel("Containment radius (deg)")

    def plot_exposure_vs_energy(self):
        """Plot exposure versus energy."""
        import matplotlib.pyplot as plt

        energy_axis = self.axes["energy_true"]
        plt.figure(figsize=(4, 3))
        plt.plot(energy_axis.center, self.exposure, color="black", lw=3)
        plt.semilogx()
        plt.xlabel(f"Energy ({energy_axis.unit})")
        plt.ylabel(f"Exposure ({self.exposure.unit})")
        plt.xlim(1e4 / 1.3, 1.3 * 1e6)
        plt.ylim(0, 1.5e11)
        plt.tight_layout()


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

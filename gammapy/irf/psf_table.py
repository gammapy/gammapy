# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.table import Table
from astropy.utils import lazyproperty
from gammapy.maps import MapAxis
from gammapy.utils.array import array_stats_str
from gammapy.utils.gauss import Gauss2DPDF
from gammapy.utils.interpolation import ScaledRegularGridInterpolator
from gammapy.utils.scripts import make_path

__all__ = ["TablePSF", "EnergyDependentTablePSF"]

log = logging.getLogger(__name__)


class TablePSF:
    """Radially-symmetric table PSF.

    Parameters
    ----------
    rad_axis : `~astropy.units.Quantity` with angle units
        Offset wrt source position
    psf_value : `~astropy.units.Quantity` with sr^-1 units
        PSF value array
    interp_kwargs : dict
        Keyword arguments passed to `ScaledRegularGridInterpolator`
    """

    def __init__(self, rad_axis, psf_value, interp_kwargs=None):
        if rad_axis.name != "rad":
            raise ValueError(
                f'rad_axis has wrong name, expected "rad", got: {rad_axis.name}'
            )

        self._rad_axis = rad_axis

        self.psf_value = u.Quantity(psf_value).to("sr^-1")

        self._interp_kwargs = interp_kwargs or {}

    @property
    def rad_axis(self):
        return self._rad_axis

    @lazyproperty
    def _interpolate(self):
        points = (self.rad_axis.center,)
        return ScaledRegularGridInterpolator(
            points=points, values=self.psf_value, **self._interp_kwargs
        )

    @lazyproperty
    def _interpolate_containment(self):
        rad_drad = (
            2 * np.pi * self.rad_axis.center * self.psf_value * self.rad_axis.bin_width
        )
        values = rad_drad.cumsum().to_value("")

        rad = self.rad_axis.edges
        values = np.insert(values, 0, 0)

        return ScaledRegularGridInterpolator(
            points=(rad,), values=values, fill_value=1,
        )

    @classmethod
    def from_shape(cls, shape, width, rad):
        """Make TablePSF objects with commonly used shapes.

        This function is mostly useful for examples and testing.

        Parameters
        ----------
        shape : {'disk', 'gauss'}
            PSF shape.
        width : `~astropy.units.Quantity` with angle units
            PSF width angle (radius for disk, sigma for Gauss).
        rad : `~astropy.units.Quantity` with angle units
            Offset angle

        Returns
        -------
        psf : `TablePSF`
            Table PSF

        Examples
        --------
        >>> import numpy as np
        >>> from astropy.coordinates import Angle
        >>> from gammapy.irf import TablePSF
        >>> rad = Angle(np.linspace(0, 0.7, 100), 'deg')
        >>> psf = TablePSF.from_shape(shape='gauss', width='0.2 deg', rad=rad)
        """
        width = Angle(width)
        rad = Angle(rad)

        if shape == "disk":
            amplitude = 1 / (np.pi * width.radian ** 2)
            psf_value = np.where(rad < width, amplitude, 0)
        elif shape == "gauss":
            gauss2d_pdf = Gauss2DPDF(sigma=width.radian)
            psf_value = gauss2d_pdf(rad.radian)
        else:
            raise ValueError(f"Invalid shape: {shape}")

        psf_value = u.Quantity(psf_value, "sr^-1")
        rad_axis = MapAxis.from_nodes(rad, name="rad")
        return cls(rad_axis=rad_axis, psf_value=psf_value)

    def info(self):
        """Print basic info."""
        ss = array_stats_str(self.rad_axis.center, "offset")
        ss += f"integral = {self.containment(self.rad_axis.edges[-1])}\n"

        for containment in [68, 80, 95]:
            radius = self.containment_radius(0.01 * containment)
            ss += f"containment radius {radius.deg} deg for {containment}%\n"

        return ss

    def evaluate(self, rad):
        r"""Evaluate PSF.

        The following PSF quantities are available:

        * 'dp_domega': PDF per 2-dim solid angle :math:`\Omega` in sr^-1

            .. math:: \frac{dP}{d\Omega}


        Parameters
        ----------
        rad : `~astropy.coordinates.Angle`
            Offset wrt source position

        Returns
        -------
        psf_value : `~astropy.units.Quantity`
            PSF value
        """
        rad = np.atleast_1d(u.Quantity(rad))
        return self._interpolate((rad,))

    def containment(self, rad_max):
        """Compute PSF containment fraction.

        Parameters
        ----------
        rad_max : `~astropy.units.Quantity`
            Offset angle range

        Returns
        -------
        integral : float
            PSF integral
        """
        rad = np.atleast_1d(rad_max)
        return self._interpolate_containment((rad,))

    def containment_radius(self, fraction):
        """Containment radius.

        Parameters
        ----------
        fraction : array_like
            Containment fraction (range 0 .. 1)

        Returns
        -------
        rad : `~astropy.coordinates.Angle`
            Containment radius angle
        """
        # TODO: check whether starting
        rad_max = Angle(
            np.linspace(0 * u.deg, self.rad_axis.center[-1], 10 * self.rad_axis.nbin),
            "rad",
        )

        containment = self.containment(rad_max=rad_max)

        fraction = np.atleast_1d(fraction)

        fraction_idx = np.argmin(np.abs(containment - fraction[:, np.newaxis]), axis=1)
        return rad_max[fraction_idx].to("deg")

    def normalize(self):
        """Normalize PSF to unit integral.

        Computes the total PSF integral via the :math:`dP / dr` spline
        and then divides the :math:`dP / dr` array.
        """
        integral = self.containment(self.rad_axis.edges[-1])
        self.psf_value /= integral

    def plot_psf_vs_rad(self, ax=None, **kwargs):
        """Plot PSF vs radius.

        Parameters
        ----------
        ax : ``

        kwargs : dict
            Keyword arguments passed to `matplotlib.pyplot.plot`
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        ax.plot(
            self.rad_axis.center.to_value("deg"),
            self.psf_value.to_value("sr-1"),
            **kwargs,
        )
        ax.set_yscale("log")
        ax.set_xlabel("Radius (deg)")
        ax.set_ylabel("PSF (sr-1)")


class EnergyDependentTablePSF:
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
    psf_value : `~astropy.units.Quantity`
        PSF (2-dim with axes: psf[energy_index, offset_index]
    interp_kwargs : dict
        Interpolation keyword arguments pass to `ScaledRegularGridInterpolator`.
    """

    def __init__(
        self,
        energy_axis_true,
        rad_axis,
        exposure=None,
        psf_value=None,
        interp_kwargs=None,
    ):
        self._rad_axis = rad_axis
        self._energy_axis_true = energy_axis_true

        assert energy_axis_true.name == "energy_true"
        assert rad_axis.name == "rad"

        if exposure is None:
            self.exposure = u.Quantity(np.ones(self.energy_axis_true.nbin), "cm^2 s")
        else:
            self.exposure = u.Quantity(exposure).to("cm^2 s")

        shape = (energy_axis_true.nbin, rad_axis.nbin)
        if psf_value is None:
            self.psf_value = np.zeros(shape) * u.Unit("sr^-1")
        else:
            if np.shape(psf_value) != shape:
                raise ValueError(
                    "psf_value has wrong shape"
                    f", expected {shape}, got {np.shape(psf_value)}"
                )
            self.psf_value = u.Quantity(psf_value).to("sr^-1")

        self._interp_kwargs = interp_kwargs or {}

    @property
    def energy_axis_true(self):
        return self._energy_axis_true

    @property
    def rad_axis(self):
        return self._rad_axis

    @lazyproperty
    def _interpolate(self):
        points = (self.energy_axis_true.center, self.rad_axis.center)
        return ScaledRegularGridInterpolator(
            points=points, values=self.psf_value, **self._interp_kwargs
        )

    @lazyproperty
    def _interpolate_containment(self):
        rad_drad = (
            2 * np.pi * self.rad_axis.center * self.psf_value * self.rad_axis.bin_width
        )
        values = rad_drad.cumsum(axis=1).to_value("")

        rad = self.rad_axis.edges
        values = np.insert(values, 0, 0, axis=1)

        points = (self.energy_axis_true.center, rad)
        return ScaledRegularGridInterpolator(
            points=points, values=values, fill_value=1,
        )

    def __str__(self):
        ss = "EnergyDependentTablePSF\n"
        ss += "-----------------------\n"
        ss += "\nAxis info:\n"
        ss += "  " + array_stats_str(self.rad_axis.center.to("deg"), "rad")
        ss += "  " + array_stats_str(self.energy_axis_true.center, "energy")
        ss += "\nContainment info:\n"
        # Print some example containment radii
        fractions = [0.68, 0.95]
        energies = u.Quantity([10, 100], "GeV")
        for fraction in fractions:
            rads = self.containment_radius(energy=energies, fraction=fraction)
            for energy, rad in zip(energies, rads):
                ss += f"  {100 * fraction}% containment radius at {energy:3.0f}: {rad:.2f}\n"

        return ss

    @classmethod
    def from_hdulist(cls, hdu_list):
        """Create `EnergyDependentTablePSF` from ``gtpsf`` format HDU list.

        Parameters
        ----------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list with ``THETA`` and ``PSF`` extensions.
        """
        # TODO: move this to MapAxis.from_table()
        rad = Angle(hdu_list["THETA"].data["Theta"], "deg")
        rad_axis = MapAxis.from_nodes(rad, name="rad")
        energy = u.Quantity(hdu_list["PSF"].data["Energy"], "MeV")
        energy_axis_true = MapAxis.from_nodes(energy, name="energy_true", interp="log")
        exposure = u.Quantity(hdu_list["PSF"].data["Exposure"], "cm^2 s")
        psf_value = u.Quantity(hdu_list["PSF"].data["PSF"], "sr^-1")

        return cls(
            energy_axis_true=energy_axis_true,
            rad_axis=rad_axis,
            exposure=exposure,
            psf_value=psf_value,
        )

    def to_hdulist(self):
        """Convert to FITS HDU list format.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            PSF in HDU list format.
        """
        # TODO: write HEADER keywords as gtpsf

        data = Table([self.rad_axis.center.to("deg")], names=["Theta"])
        theta_hdu = fits.BinTableHDU(data=data, name="THETA")

        data = Table(
            [
                self.energy_axis_true.center.to("MeV"),
                self.exposure.to("cm^2 s"),
                self.psf_value.to("sr^-1"),
            ],
            names=["Energy", "Exposure", "PSF"],
        )
        psf_hdu = fits.BinTableHDU(data=data, name="PSF")

        hdu_list = fits.HDUList([fits.PrimaryHDU(), theta_hdu, psf_hdu])
        return hdu_list

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

    def evaluate(self, energy=None, rad=None, method="linear"):
        """Evaluate the PSF at a given energy and offset

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy value
        rad : `~astropy.coordinates.Angle`
            Offset wrt source position
        method : {"linear", "nearest"}
            Linear or nearest neighbour interpolation.

        Returns
        -------
        values : `~astropy.units.Quantity`
            Interpolated value
        """
        if energy is None:
            energy = self.energy_axis_true.center

        if rad is None:
            rad = self.rad_axis.center

        energy = u.Quantity(energy, ndmin=1)[:, np.newaxis]
        rad = u.Quantity(rad, ndmin=1)
        return self._interpolate((energy, rad), clip=True, method=method)

    def table_psf_at_energy(self, energy, method="linear", **kwargs):
        """Create `~gammapy.irf.TablePSF` at one given energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        method : {"linear", "nearest"}
            Linear or nearest neighbour interpolation.

        Returns
        -------
        psf : `~gammapy.irf.TablePSF`
            Table PSF
        """
        psf_value = self.evaluate(energy=energy, method=method)[0, :]
        return TablePSF(rad_axis=self.rad_axis, psf_value=psf_value, **kwargs)

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
            weigthed PSF.

        Returns
        -------
        psf : `TablePSF`
            Table PSF
        """
        from gammapy.modeling.models import PowerLawSpectralModel, TemplateSpectralModel

        if spectrum is None:
            spectrum = PowerLawSpectralModel()

        exposure = TemplateSpectralModel(self.energy_axis_true.center, self.exposure)

        e_min, e_max = energy_range
        energy = MapAxis.from_energy_bounds(e_min, e_max, n_bins).edges

        weights = spectrum(energy) * exposure(energy)
        weights /= weights.sum()

        psf_value = self.evaluate(energy=energy)
        psf_value_weighted = weights[:, np.newaxis] * psf_value
        return TablePSF(self.rad_axis, psf_value_weighted.sum(axis=0), **kwargs)

    def containment_radius(self, energy, fraction=0.68):
        """Containment radius.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        fraction : float
            Containment fraction.

        Returns
        -------
        rad : `~astropy.units.Quantity`
            Containment radius in deg
        """
        # upsamle for better precision
        rad_max = Angle(self.rad_axis.upsample(factor=10).center)
        containment = self.containment(energy=energy, rad_max=rad_max)

        # find nearest containment value
        fraction_idx = np.argmin(np.abs(containment - fraction), axis=1)
        return rad_max[fraction_idx].to("deg")

    def containment(self, energy, rad_max):
        """Compute containment of the PSF.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        rad_max : `~astropy.coordinates.Angle`
            Maximum offset angle.

        Returns
        -------
        fraction : array_like
            Containment fraction (in range 0 .. 1)
        """
        energy = np.atleast_1d(u.Quantity(energy))[:, np.newaxis]
        rad_max = np.atleast_1d(u.Quantity(rad_max))
        return self._interpolate_containment((energy, rad_max))

    def info(self):
        """Print basic info"""
        print(str(self))

    def plot_psf_vs_rad(self, energies=None, ax=None, **kwargs):
        """Plot PSF vs radius.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energies where to plot the PSF.
        **kwargs : dict
            Keyword arguments pass to `~matplotlib.pyplot.plot`.
        """
        import matplotlib.pyplot as plt

        if energies is None:
            energies = [100, 1000, 10000] * u.GeV

        ax = plt.gca() if ax is None else ax

        for energy in energies:
            psf_value = np.squeeze(self.evaluate(energy=energy))
            label = f"{energy:.0f}"
            ax.plot(
                self.rad_axis.center.to_value("deg"),
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

        for fraction in fractions:
            rad = self.containment_radius(self.energy_axis_true.center, fraction)
            label = f"{100 * fraction:.1f}% Containment"
            ax.plot(
                self.energy_axis_true.center.to("GeV").value,
                rad.to("deg").value,
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

        plt.figure(figsize=(4, 3))
        plt.plot(self.energy_axis_true.center, self.exposure, color="black", lw=3)
        plt.semilogx()
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Exposure (cm^2 s)")
        plt.xlim(1e4 / 1.3, 1.3 * 1e6)
        plt.ylim(0, 1.5e11)
        plt.tight_layout()

    def stack(self, psf):
        """Stack two EnergyDependentTablePSF objects.s

        Parameters
        ----------
        psf : `EnergyDependentTablePSF`
            PSF to stack.

        Returns
        -------
        stacked_psf : `EnergyDependentTablePSF`
            Stacked PSF.

        """
        exposure = self.exposure + psf.exposure
        psf_value = self.psf_value.T * self.exposure + psf.psf_value.T * psf.exposure

        with np.errstate(invalid="ignore"):
            # exposure can be zero
            psf_value = np.nan_to_num(psf_value / exposure)

        return self.__class__(
            energy_axis_true=self.energy_axis_true,
            rad_axis=self.rad_axis,
            psf_value=psf_value.T,
            exposure=exposure,
        )

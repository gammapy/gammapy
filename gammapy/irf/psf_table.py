# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
import scipy.integrate
from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.utils import lazyproperty
from gammapy.utils.array import array_stats_str
from gammapy.utils.energy import energy_logspace
from gammapy.utils.gauss import Gauss2DPDF
from gammapy.utils.interpolation import ScaledRegularGridInterpolator
from gammapy.utils.scripts import make_path

__all__ = ["TablePSF", "EnergyDependentTablePSF"]

log = logging.getLogger(__name__)


class TablePSF:
    """Radially-symmetric table PSF.

    Parameters
    ----------
    rad : `~astropy.units.Quantity` with angle units
        Offset wrt source position
    psf_value : `~astropy.units.Quantity` with sr^-1 units
        PSF value array
    interp_kwargs : dict
        Keyword arguments passed to `ScaledRegularGridInterpolator`
    """

    def __init__(self, rad, psf_value, interp_kwargs=None):
        self.rad = Angle(rad).to("rad")
        self.psf_value = u.Quantity(psf_value).to("sr^-1")

        self._interp_kwargs = interp_kwargs or {}

    @lazyproperty
    def _interpolate(self):
        points = (self.rad,)
        return ScaledRegularGridInterpolator(
            points=points, values=self.psf_value, **self._interp_kwargs
        )

    @lazyproperty
    def _interpolate_containment(self):
        if self.rad[0] > 0:
            rad = self.rad.insert(0, 0)
        else:
            rad = self.rad

        rad_drad = 2 * np.pi * rad * self.evaluate(rad)
        values = scipy.integrate.cumtrapz(
            rad_drad.to_value("rad-1"), rad.to_value("rad"), initial=0
        )

        return ScaledRegularGridInterpolator(points=(rad,), values=values, fill_value=1)

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

        return cls(rad, psf_value)

    def info(self):
        """Print basic info."""
        ss = array_stats_str(self.rad.deg, "offset")
        ss += f"integral = {self.integral()}\n"

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
        rad_max = Angle(np.linspace(0, self.rad[-1].value, 10 * len(self.rad)), "rad")
        containment = self.containment(rad_max=rad_max)

        if not np.allclose(containment.max(), 1, atol=0.01):
            log.warn(
                "PSF does not integrate to unity within a precision of 1%."
                " Containment radius computation might give biased results."
            )

        fraction = np.atleast_1d(fraction)

        fraction_idx = np.argmin(np.abs(containment - fraction[:, np.newaxis]), axis=1)
        return rad_max[fraction_idx].to("deg")

    def normalize(self):
        """Normalize PSF to unit integral.

        Computes the total PSF integral via the :math:`dP / dr` spline
        and then divides the :math:`dP / dr` array.
        """
        integral = self.containment(self.rad[-1])
        self.psf_value /= integral

    def broaden(self, factor, normalize=True):
        r"""Broaden PSF by scaling the offset array.

        For a broadening factor :math:`f` and the offset
        array :math:`r`, the offset array scaled
        in the following way:

        .. math::
            r_{new} = f \times r_{old}
            \frac{dP}{dr}(r_{new}) = \frac{dP}{dr}(r_{old})

        Parameters
        ----------
        factor : float
            Broadening factor
        normalize : bool
            Normalize PSF after broadening
        """
        self.rad *= factor
        self._setup_interpolators()
        if normalize:
            self.normalize()

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

        ax.plot(self.rad.to_value("deg"), self.psf_value.to_value("sr-1"), **kwargs)
        ax.set_yscale("log")
        ax.set_xlabel("Radius (deg)")
        ax.set_ylabel("PSF (sr-1)")


class EnergyDependentTablePSF:
    """Energy-dependent radially-symmetric table PSF (``gtpsf`` format).

    TODO: add references and explanations.

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Energy (1-dim)
    rad : `~astropy.units.Quantity` with angle units
        Offset angle wrt source position (1-dim)
    exposure : `~astropy.units.Quantity`
        Exposure (1-dim)
    psf_value : `~astropy.units.Quantity`
        PSF (2-dim with axes: psf[energy_index, offset_index]
    interp_kwargs : dict
        Interpolation keyword arguments pass to `ScaledRegularGridInterpolator`.
    """

    def __init__(self, energy, rad, exposure=None, psf_value=None, interp_kwargs=None):
        self.energy = u.Quantity(energy).to("GeV")
        self.rad = u.Quantity(rad).to("radian")
        if exposure is None:
            self.exposure = u.Quantity(np.ones(len(energy)), "cm^2 s")
        else:
            self.exposure = u.Quantity(exposure).to("cm^2 s")

        if psf_value is None:
            self.psf_value = u.Quantity(np.zeros(len(energy), len(rad)), "sr^-1")
        else:
            self.psf_value = u.Quantity(psf_value).to("sr^-1")

        self._interp_kwargs = interp_kwargs or {}

    @lazyproperty
    def _interpolate(self):
        points = (self.energy, self.rad)
        return ScaledRegularGridInterpolator(
            points=points, values=self.psf_value, **self._interp_kwargs
        )

    @lazyproperty
    def _interpolate_containment(self):
        if self.rad[0] > 0:
            rad = self.rad.insert(0, 0)
        else:
            rad = self.rad

        rad_drad = 2 * np.pi * rad * self.evaluate(energy=self.energy, rad=rad)
        values = scipy.integrate.cumtrapz(
            rad_drad.to_value("rad-1"), rad.to_value("rad"), initial=0, axis=1
        )

        points = (self.energy, rad)
        return ScaledRegularGridInterpolator(points=points, values=values, fill_value=1)

    def __str__(self):
        ss = "EnergyDependentTablePSF\n"
        ss += "-----------------------\n"
        ss += "\nAxis info:\n"
        ss += "  " + array_stats_str(self.rad.to("deg"), "rad")
        ss += "  " + array_stats_str(self.energy, "energy")
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
    def from_fits(cls, hdu_list):
        """Create `EnergyDependentTablePSF` from ``gtpsf`` format HDU list.

        Parameters
        ----------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list with ``THETA`` and ``PSF`` extensions.
        """
        rad = Angle(hdu_list["THETA"].data["Theta"], "deg")
        energy = u.Quantity(hdu_list["PSF"].data["Energy"], "MeV")
        exposure = u.Quantity(hdu_list["PSF"].data["Exposure"], "cm^2 s")
        psf_value = u.Quantity(hdu_list["PSF"].data["PSF"], "sr^-1")

        return cls(energy, rad, exposure, psf_value)

    def to_fits(self):
        """Convert to FITS HDU list format.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            PSF in HDU list format.
        """
        # TODO: write HEADER keywords as gtpsf

        data = self.rad
        theta_hdu = fits.BinTableHDU(data=data, name="Theta")

        data = [self.energy, self.exposure, self.psf_value]
        psf_hdu = fits.BinTableHDU(data=data, name="PSF")

        hdu_list = fits.HDUList([theta_hdu, psf_hdu])
        return hdu_list

    @classmethod
    def read(cls, filename):
        """Create `EnergyDependentTablePSF` from ``gtpsf``-format FITS file.

        Parameters
        ----------
        filename : str
            File name
        """
        with fits.open(make_path(filename), memmap=False) as hdulist:
            return cls.from_fits(hdulist)

    def write(self, *args, **kwargs):
        """Write to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_fits().writeto(*args, **kwargs)

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
            energy = self.energy

        if rad is None:
            rad = self.rad

        energy = np.atleast_1d(u.Quantity(energy))[:, np.newaxis]
        rad = np.atleast_1d(u.Quantity(rad))
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
        return TablePSF(self.rad, psf_value, **kwargs)

    def table_psf_in_energy_band(self, energy_band, spectrum=None, n_bins=11, **kwargs):
        """Average PSF in a given energy band.

        Expected counts in sub energy bands given the given exposure
        and spectrum are used as weights.

        Parameters
        ----------
        energy_band : `~astropy.units.Quantity`
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

        exposure = TemplateSpectralModel(self.energy, self.exposure)

        e_min, e_max = energy_band
        energy = energy_logspace(emin=e_min, emax=e_max, nbins=n_bins)

        weights = spectrum(energy) * exposure(energy)
        weights /= weights.sum()

        psf_value = self.evaluate(energy=energy)
        psf_value_weighted = weights[:, np.newaxis] * psf_value
        return TablePSF(self.rad, psf_value_weighted.sum(axis=0), **kwargs)

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
        rad_max = Angle(np.linspace(0, self.rad[-1].value, 10 * len(self.rad)), "rad")
        containment = self.containment(energy=energy, rad_max=rad_max)

        if not np.allclose(containment.max(axis=1), 1, atol=0.01):
            log.warning(
                "PSF does not integrate to unity within a precision of 1% in each energy bin."
                " Containment radius computation might give biased results."
            )

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
                self.rad.to_value("deg"),
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
            rad = self.containment_radius(self.energy, fraction)
            label = f"{100 * fraction:.1f}% Containment"
            ax.plot(self.energy.value, rad.value, label=label, **kwargs)

        ax.semilogx()
        ax.legend(loc="best")
        ax.set_xlabel("Energy (GeV)")
        ax.set_ylabel("Containment radius (deg)")

    def plot_exposure_vs_energy(self):
        """Plot exposure versus energy."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 3))
        plt.plot(self.energy, self.exposure, color="black", lw=3)
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
            energy=self.energy, rad=self.rad, psf_value=psf_value.T, exposure=exposure
        )

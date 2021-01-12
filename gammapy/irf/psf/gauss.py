# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.table import Table
from gammapy.maps import MapAxes, MapAxis
from gammapy.utils.array import array_stats_str
from gammapy.utils.gauss import MultiGauss2D
from gammapy.utils.interpolation import ScaledRegularGridInterpolator
from gammapy.utils.scripts import make_path
from .table import PSF3D, EnergyDependentTablePSF

__all__ = ["EnergyDependentMultiGaussPSF"]

log = logging.getLogger(__name__)


class EnergyDependentMultiGaussPSF:
    """Triple Gauss analytical PSF depending on energy and theta.

    To evaluate the PSF call the ``to_energy_dependent_table_psf`` or ``psf_at_energy_and_theta`` methods.

    Parameters
    ----------
    energy_axis_true : `MapAxis`
        True energy axis
    offset_axis : `MapAxis`
        Offset axis.
    sigmas : list of 'numpy.ndarray'
        Triple Gauss sigma parameters, where every entry is
        a two dimensional 'numpy.ndarray' containing the sigma
        value for every given energy and theta.
    norms : list of 'numpy.ndarray'
        Triple Gauss norm parameters, where every entry is
        a two dimensional 'numpy.ndarray' containing the norm
        value for every given energy and theta. Norm corresponds
        to the value of the Gaussian at theta = 0.
    meta : dict
        Meta data

    Examples
    --------
    Plot R68 of the PSF vs. theta and energy:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from gammapy.irf import EnergyDependentMultiGaussPSF
        filename = '$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits'
        psf = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
        psf.plot_containment(0.68)
        plt.show()
    """

    tag = "psf_3gauss"

    def __init__(
        self,
        energy_axis_true,
        offset_axis,
        sigmas,
        norms,
        meta,
    ):
        energy_axis_true.assert_name("energy_true")
        offset_axis.assert_name("offset")

        self._energy_axis_true = energy_axis_true
        self._offset_axis = offset_axis

        sigmas[0][sigmas[0] == 0] = 1
        sigmas[1][sigmas[1] == 0] = 1
        sigmas[2][sigmas[2] == 0] = 1
        self.sigmas = sigmas

        self.norms = norms
        self.meta = meta or {}
        self._interp_norms = self._setup_interpolators(self.norms)
        self._interp_sigmas = self._setup_interpolators(self.sigmas)

    @property
    def energy_thresh_lo(self):
        """Low energy threshold"""
        return self.meta["LO_THRES"] * u.TeV

    @property
    def energy_thresh_hi(self):
        """High energy threshold"""
        return self.meta["HI_THRES"] * u.TeV

    @property
    def energy_axis_true(self):
        return self._energy_axis_true

    @property
    def offset_axis(self):
        return self._offset_axis

    def _setup_interpolators(self, values_list):
        interps = []
        for values in values_list:
            interp = ScaledRegularGridInterpolator(
                points=(self.offset_axis.center, self.energy_axis_true.center),
                values=values,
            )
            interps.append(interp)
        return interps

    @classmethod
    def read(cls, filename, hdu="PSF_2D_GAUSS"):
        """Create `EnergyDependentMultiGaussPSF` from FITS file.

        Parameters
        ----------
        filename : str
            File name
        """
        with fits.open(str(make_path(filename)), memmap=False) as hdulist:
            return cls.from_table_hdu(hdulist[hdu])

    @classmethod
    def from_table_hdu(cls, hdu):
        """Create `EnergyDependentMultiGaussPSF` from HDU list.

        Parameters
        ----------
        hdu : `~astropy.io.fits.BinTableHDU`
            HDU
        """
        table = Table.read(hdu)

        energy_axis_true = MapAxis.from_table(
            table, column_prefix="ENERG", format="gadf-dl3"
        )
        offset_axis = MapAxis.from_table(
            table, column_prefix="THETA", format="gadf-dl3"
        )

        # Get sigmas
        shape = (offset_axis.nbin, energy_axis_true.nbin)
        sigmas = []
        for key in ["SIGMA_1", "SIGMA_2", "SIGMA_3"]:
            sigma = hdu.data[key].reshape(shape).copy()
            sigmas.append(sigma)

        # Get amplitudes
        norms = []
        for key in ["SCALE", "AMPL_2", "AMPL_3"]:
            norm = hdu.data[key].reshape(shape).copy()
            norms.append(norm)

        return cls(
            energy_axis_true=energy_axis_true,
            offset_axis=offset_axis,
            sigmas=sigmas,
            norms=norms,
            meta=dict(hdu.header)
        )

    def to_hdulist(self):
        """
        Convert psf table data to FITS hdu list.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            PSF in HDU list format.
        """
        # Set up data
        names = [
            "SCALE",
            "SIGMA_1",
            "AMPL_2",
            "SIGMA_2",
            "AMPL_3",
            "SIGMA_3",
        ]
        units = ["", "deg", "", "deg", "", "deg"]

        data = [
            self.norms[0],
            self.sigmas[0],
            self.norms[1],
            self.sigmas[1],
            self.norms[2],
            self.sigmas[2],
        ]

        axes = MapAxes([self.energy_axis_true, self.offset_axis])
        table = axes.to_table(format="gadf-dl3")

        for name_, data_, unit_ in zip(names, data, units):
            table[name_] = [data_]
            table[name_].unit = unit_

        # Create hdu and hdu list
        hdu = fits.BinTableHDU(table)
        hdu.header.update(self.meta)
        return fits.HDUList([fits.PrimaryHDU(), hdu])

    def write(self, filename, *args, **kwargs):
        """Write PSF to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_hdulist().writeto(str(make_path(filename)), *args, **kwargs)

    def psf_at_energy_and_theta(self, energy, theta):
        """
        Get `~gammapy.modeling.models.MultiGauss2D` model for given energy and theta.

        No interpolation is used.

        Parameters
        ----------
        energy : `~astropy.units.u.Quantity`
            Energy at which a PSF is requested.
        theta : `~astropy.coordinates.Angle`
            Offset angle at which a PSF is requested.

        Returns
        -------
        psf : `~gammapy.utils.gauss.MultiGauss2D`
            Multigauss PSF object.
        """
        energy = u.Quantity(energy)
        theta = u.Quantity(theta)

        sigmas, norms = [], []

        pars = {"A_1": 1}

        for interp_sigma in self._interp_sigmas:
            sigma = interp_sigma((theta, energy))
            sigmas.append(sigma)

        for name, interp_norm in zip(["scale", "A_2", "A_3"], self._interp_norms):
            pars[name] = interp_norm((theta, energy))

        for idx, sigma in enumerate(sigmas):
            a = pars[f"A_{idx + 1}"]
            norm = pars["scale"] * 2 * a * sigma ** 2
            norms.append(norm)

        m = MultiGauss2D(sigmas, norms)
        m.normalize()
        return m

    def containment_radius(self, energy, theta, fraction=0.68):
        """Compute containment for all energy and theta values"""
        # This is a false positive from pylint
        # See https://github.com/PyCQA/pylint/issues/2435
        energies = u.Quantity(
            energy
        ).flatten()  # pylint:disable=assignment-from-no-return
        thetas = Angle(theta).flatten()
        radius = np.empty((theta.size, energy.size))

        for idx, energy in enumerate(energies):
            for jdx, theta in enumerate(thetas):
                try:
                    psf = self.psf_at_energy_and_theta(energy, theta)
                    radius[jdx, idx] = psf.containment_radius(fraction)
                except ValueError:
                    log.debug(
                        f"Computing containment failed for energy = {energy:.2f}"
                        f" and theta={theta:.2f}"
                    )
                    log.debug(f"Sigmas: {psf.sigmas} Norms: {psf.norms}")
                    radius[jdx, idx] = np.nan
        return Angle(radius, "deg")

    def plot_containment(self, fraction=0.68, ax=None, add_cbar=True, **kwargs):
        """
        Plot containment image with energy and theta axes.

        Parameters
        ----------
        fraction : float
            Containment fraction between 0 and 1.
        add_cbar : bool
            Add a colorbar
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        energy = self.energy_axis_true.center
        offset = self.offset_axis.center

        # Set up and compute data
        containment = self.containment_radius(energy, offset, fraction)

        # plotting defaults
        kwargs.setdefault("cmap", "GnBu")
        kwargs.setdefault("vmin", np.nanmin(containment.value))
        kwargs.setdefault("vmax", np.nanmax(containment.value))

        # Plotting
        x = energy.value
        y = offset.value
        caxes = ax.pcolormesh(x, y, containment.value, **kwargs)

        # Axes labels and ticks, colobar
        ax.semilogx()
        ax.set_ylabel(f"Offset ({offset.unit})")
        ax.set_xlabel(f"Energy ({energy.unit})")
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

        try:
            self._plot_safe_energy_range(ax)
        except KeyError:
            pass

        if add_cbar:
            label = f"Containment radius R{100 * fraction:.0f} ({containment.unit})"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        return ax

    def _plot_safe_energy_range(self, ax):
        """add safe energy range lines to the plot"""
        esafe = self.energy_thresh_lo
        omin = self.offset_axis.center.min()
        omax = self.offset_axis.center.max()
        ax.vlines(x=esafe.value, ymin=omin.value, ymax=omax.value)
        label = f"Safe energy threshold: {esafe:3.2f}"
        ax.text(x=1.1 * esafe.value, y=0.3, s=label, va="top")

    def plot_containment_vs_energy(
        self, fractions=[0.68, 0.95], thetas=Angle([0, 1], "deg"), ax=None, **kwargs
    ):
        """Plot containment fraction as a function of energy.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        energy = self.energy_axis_true.center

        for theta in thetas:
            for fraction in fractions:
                radius = self.containment_radius(energy, theta, fraction).squeeze()
                kwargs.setdefault("label", f"{theta.deg} deg, {100 * fraction:.1f}%")
                ax.plot(energy.value, radius.value, **kwargs)

        ax.semilogx()
        ax.legend(loc="best")
        ax.set_xlabel("Energy (TeV)")
        ax.set_ylabel("Containment radius (deg)")

    def peek(self, figsize=(15, 5)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)

        self.plot_containment(fraction=0.68, ax=axes[0])
        self.plot_containment(fraction=0.95, ax=axes[1])
        self.plot_containment_vs_energy(ax=axes[2])

        # TODO: implement this plot
        # psf = self.psf_at_energy_and_theta(energy='1 TeV', theta='1 deg')
        # psf.plot_components(ax=axes[2])

        plt.tight_layout()

    def info(
        self,
        fractions=[0.68, 0.95],
        energies=u.Quantity([1.0, 10.0], "TeV"),
        thetas=u.Quantity([0.0], "deg"),
    ):
        """
        Print PSF summary info.

        The containment radius for given fraction, energies and thetas is
        computed and printed on the command line.

        Parameters
        ----------
        fractions : list
            Containment fraction to compute containment radius for.
        energies : `~astropy.units.u.Quantity`
            Energies to compute containment radius for.
        thetas : `~astropy.units.u.Quantity`
            Thetas to compute containment radius for.

        Returns
        -------
        ss : string
            Formatted string containing the summary info.
        """
        ss = "\nSummary PSF info\n"
        ss += "----------------\n"
        ss += array_stats_str(self.offset_axis.center.to("deg"), "Theta")
        ss += array_stats_str(self.energy_axis_true.edges[1:], "Energy hi")
        ss += array_stats_str(self.energy_axis_true.edges[:-1], "Energy lo")
        ss += f"Safe energy threshold lo: {self.energy_thresh_lo:6.3f}\n"
        ss += f"Safe energy threshold hi: {self.energy_thresh_hi:6.3f}\n"

        for fraction in fractions:
            containment = self.containment_radius(energies, thetas, fraction)
            for i, energy in enumerate(energies):
                for j, theta in enumerate(thetas):
                    radius = containment[j, i]
                    ss += (
                        "{:2.0f}% containment radius at theta = {} and "
                        "E = {:4.1f}: {:5.8f}\n"
                        "".format(100 * fraction, theta, energy, radius)
                    )
        return ss

    def to_energy_dependent_table_psf(self, theta=None, rad=None, exposure=None):
        """Convert triple Gaussian PSF ot table PSF.

        Parameters
        ----------
        theta : `~astropy.coordinates.Angle`
            Offset in the field of view. Default theta = 0 deg
        rad : `~astropy.coordinates.Angle`
            Offset from PSF center used for evaluating the PSF on a grid.
            Default offset = [0, 0.005, ..., 1.495, 1.5] deg.
        exposure : `~astropy.units.u.Quantity`
            Energy dependent exposure. Should be in units equivalent to 'cm^2 s'.
            Default exposure = 1.

        Returns
        -------
        tabe_psf : `~gammapy.irf.EnergyDependentTablePSF`
            Instance of `EnergyDependentTablePSF`.
        """
        # Convert energies to log center
        energies = self.energy_axis_true.center
        # Defaults and input handling
        if theta is None:
            theta = Angle(0, "deg")
        else:
            theta = Angle(theta)

        if rad is None:
            rad = Angle(np.arange(0, 1.5, 0.005), "deg")

        rad_axis = MapAxis.from_nodes(rad, name="rad")

        psf_value = u.Quantity(np.zeros((energies.size, rad.size)), "deg^-2")

        for idx, energy in enumerate(energies):
            psf_gauss = self.psf_at_energy_and_theta(energy, theta)
            psf_value[idx] = u.Quantity(psf_gauss(rad), "deg^-2")

        return EnergyDependentTablePSF(
            axes=[self.energy_axis_true, rad_axis],
            exposure=exposure,
            data=psf_value.value,
            unit=psf_value.unit
        )

    def to_psf3d(self, rad=None):
        """Create a PSF3D from an analytical PSF.

        Parameters
        ----------
        rad : `~astropy.units.u.Quantity` or `~astropy.coordinates.Angle`
            the array of position errors (rad) on which the PSF3D will be defined

        Returns
        -------
        psf3d : `~gammapy.irf.PSF3D`
            the PSF3D. It will be defined on the same energy and offset values than the input psf.
        """
        offsets = self.offset_axis.center
        energy = self.energy_axis_true.center

        if rad is None:
            rad = np.linspace(0, 0.66, 67) * u.deg

        rad_axis = MapAxis.from_edges(rad, name="rad")

        shape = (self.energy_axis_true.nbin, self.offset_axis.nbin, rad_axis.nbin)
        psf_value = np.zeros(shape) * u.Unit("sr-1")

        for idx, offset in enumerate(offsets):
            table_psf = self.to_energy_dependent_table_psf(offset)
            psf_value[:, idx, :] = table_psf.evaluate(energy, rad_axis.center)

        return PSF3D(
            axes=[self.energy_axis_true, self.offset_axis, rad_axis],
            data=psf_value.value,
            unit=psf_value.unit,
            meta=self.meta.copy()
        )

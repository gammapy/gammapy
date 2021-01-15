# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from gammapy.maps import MapAxis
from gammapy.utils.gauss import MultiGauss2D
from .table import PSF3D, EnergyDependentTablePSF
from .core import ParametricPSF


__all__ = ["EnergyDependentMultiGaussPSF"]

log = logging.getLogger(__name__)


class EnergyDependentMultiGaussPSF(ParametricPSF):
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
    required_axes = ["energy_true", "offset"]
    par_names = ["SIGMA_1", "SIGMA_2", "SIGMA_3", "SCALE", "AMPL_2", "AMPL_3"]
    par_units = ["deg", "deg", "deg", "", "", ""]

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

        pars = {"AMPL_1": 1}

        for name in ["SIGMA_1", "SIGMA_2", "SIGMA_3"]:
            interp = self._interpolators[name]
            sigmas.append(interp((energy, theta)))

        for name in ["SCALE", "AMPL_2", "AMPL_3"]:
            interp = self._interpolators[name]
            pars[name] = interp((energy, theta))

        for idx, sigma in enumerate(sigmas):
            a = pars[f"AMPL_{idx + 1}"]
            norm = pars["SCALE"] * 2 * a * sigma ** 2
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

        energy = self.axes["energy_true"].center
        offset = self.axes["offset"].center

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

        if add_cbar:
            label = f"Containment radius R{100 * fraction:.0f} ({containment.unit})"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        return ax

    def plot_containment_vs_energy(
        self, fractions=[0.68, 0.95], thetas=Angle([0, 1], "deg"), ax=None, **kwargs
    ):
        """Plot containment fraction as a function of energy.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        energy = self.axes["energy_true"].center

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

    def to_energy_dependent_table_psf(self, theta=None, rad=None):
        """Convert triple Gaussian PSF ot table PSF.

        Parameters
        ----------
        theta : `~astropy.coordinates.Angle`
            Offset in the field of view. Default theta = 0 deg
        rad : `~astropy.coordinates.Angle`
            Offset from PSF center used for evaluating the PSF on a grid.
            Default offset = [0, 0.005, ..., 1.495, 1.5] deg.

        Returns
        -------
        tabe_psf : `~gammapy.irf.EnergyDependentTablePSF`
            Instance of `EnergyDependentTablePSF`.
        """
        # Convert energies to log center
        energy_axis_true = self.axes["energy_true"]

        # Defaults and input handling
        if theta is None:
            theta = Angle(0, "deg")
        else:
            theta = Angle(theta)

        if rad is None:
            rad = Angle(np.arange(0, 1.5, 0.005), "deg")

        rad_axis = MapAxis.from_nodes(rad, name="rad")

        psf_value = u.Quantity(np.zeros((energy_axis_true.nbin, rad.size)), "deg^-2")

        for idx, energy in enumerate(energy_axis_true.center):
            psf_gauss = self.psf_at_energy_and_theta(energy, theta)
            psf_value[idx] = u.Quantity(psf_gauss(rad), "deg^-2")

        return EnergyDependentTablePSF(
            axes=[energy_axis_true, rad_axis],
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
        offset_axis = self.axes["offset"]
        energy_axis_true = self.axes["energy_true"]

        if rad is None:
            rad = np.linspace(0, 0.66, 67) * u.deg

        rad_axis = MapAxis.from_edges(rad, name="rad")

        shape = (energy_axis_true.nbin, offset_axis.nbin, rad_axis.nbin)
        psf_value = np.zeros(shape) * u.Unit("sr-1")

        for idx, offset in enumerate(offset_axis.center):
            table_psf = self.to_energy_dependent_table_psf(offset)
            psf_value[:, idx, :] = table_psf.evaluate(
                energy_true=energy_axis_true.center[:, np.newaxis], rad=rad_axis.center
            )

        return PSF3D(
            axes=[energy_axis_true, offset_axis, rad_axis],
            data=psf_value.value,
            unit=psf_value.unit,
            meta=self.meta.copy()
        )

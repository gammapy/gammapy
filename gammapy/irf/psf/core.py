# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy import units as u
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from gammapy.utils.array import array_stats_str
from ..core import IRF


class PSF(IRF):
    """PSF base class"""

    def normalize(self):
        """Normalise psf"""
        super().normalize(axis_name="rad")

    def containment(self, rad, **kwargs):
        """Containment tof the PSF at given axes coordinates

        Parameters
        ----------
        rad : `~astropy.units.Quantity`
            Rad value
        **kwargs : dict
            Other coordinates

        Returns
        -------
        containment : `~numpy.ndarray`
            Containment
        """
        containment = self.integral(axis_name="rad", rad=rad, **kwargs)
        return containment.to("")

    def containment_radius(self, fraction, factor=20, **kwargs):
        """Containment radius at given axes coordinates

        Parameters
        ----------
        fraction : float or `~numpy.ndarray`
            Containment fraction
        factor : int
            Up-sampling factor of the rad axis, determines the precision of the
            computed containment radius.
        **kwargs : dict
            Other coordinates

        Returns
        -------
        radius : `~astropy.coordinates.Angle`
            Containment radius
        """
        # TODO: this uses a lot of numpy broadcasting tricks, maybe simplify...
        from gammapy.datasets.map import RAD_AXIS_DEFAULT

        output = np.broadcast(*kwargs.values(), fraction)

        try:
            rad_axis = self.axes["rad"]
        except KeyError:
            rad_axis = RAD_AXIS_DEFAULT

        # upsample for better precision
        rad = rad_axis.upsample(factor=factor).center

        axis = tuple(range(output.ndim))
        rad = np.expand_dims(rad, axis=axis).T
        containment = self.containment(rad=rad, **kwargs)

        fraction_idx = np.argmin(np.abs(containment - fraction), axis=0)
        return rad[fraction_idx].reshape(output.shape)

    def info(
        self,
        fraction=(0.68, 0.95),
        energy_true=([1.0], [10.0]) * u.TeV,
        offset=0 * u.deg,
    ):
        """
        Print PSF summary info.

        The containment radius for given fraction, energies and thetas is
        computed and printed on the command line.

        Parameters
        ----------
        fraction : list
            Containment fraction to compute containment radius for.
        energy_true : `~astropy.units.u.Quantity`
            Energies to compute containment radius for.
        offset : `~astropy.units.u.Quantity`
            Offset to compute containment radius for.

        Returns
        -------
        ss : string
            Formatted string containing the summary info.
        """
        info = "\nSummary PSF info\n"
        info += "----------------\n"
        info += array_stats_str(self.axes["offset"].center.to("deg"), "Theta")
        info += array_stats_str(self.axes["energy_true"].edges[1:], "Energy hi")
        info += array_stats_str(self.axes["energy_true"].edges[:-1], "Energy lo")

        containment_radius = self.containment_radius(
            energy_true=energy_true, offset=offset, fraction=fraction
        )

        energy_true, offset, fraction = np.broadcast_arrays(
            energy_true, offset, fraction, subok=True
        )

        for idx in np.ndindex(containment_radius.shape):
            info += f"{100 * fraction[idx]:.2f} containment radius "
            info += f"at offset = {offset[idx]} "
            info += f"and energy_true = {energy_true[idx]:4.1f}: "
            info += f"{containment_radius[idx]:.3f}\n"

        return info

    def plot_containment_radius_vs_energy(
        self, ax=None, fraction=(0.68, 0.95), offset=(0, 1) * u.deg, **kwargs
    ):
        """Plot containment fraction as a function of energy.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Axes to plot on.
        fraction : list of float or `~numpy.ndarray`
            Containment fraction between 0 and 1.
        offset : `~astropy.units.Quantity`
            Offset array
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.plot`

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
             Axes to plot on.

        """
        ax = plt.gca() if ax is None else ax

        energy_true = self.axes["energy_true"]

        for theta in offset:
            for frac in fraction:
                plot_kwargs = kwargs.copy()
                radius = self.containment_radius(
                    energy_true=energy_true.center, offset=theta, fraction=frac
                )
                plot_kwargs.setdefault("label", f"{theta}, {100 * frac:.1f}%")
                with quantity_support():
                    ax.plot(energy_true.center, radius, **plot_kwargs)

        energy_true.format_plot_xaxis(ax=ax)
        ax.legend(loc="best")
        ax.set_ylabel(f"Containment radius [{ax.yaxis.units}]")
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
        return ax

    def plot_containment_radius(self, ax=None, fraction=0.68, add_cbar=True, **kwargs):
        """Plot containment image with energy and theta axes.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Axes to plot on.
        fraction : float
            Containment fraction between 0 and 1.
        add_cbar : bool
            Add a colorbar
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.pcolormesh`

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
             Axes to plot on.
        """
        ax = plt.gca() if ax is None else ax

        energy = self.axes["energy_true"]
        offset = self.axes["offset"]

        # Set up and compute data
        containment = self.containment_radius(
            energy_true=energy.center[:, np.newaxis],
            offset=offset.center,
            fraction=fraction,
        )

        # plotting defaults
        kwargs.setdefault("cmap", "GnBu")
        kwargs.setdefault("vmin", np.nanmin(containment.value))
        kwargs.setdefault("vmax", np.nanmax(containment.value))

        # Plotting
        with quantity_support():
            caxes = ax.pcolormesh(
                energy.edges, offset.edges, containment.value.T, **kwargs
            )

        energy.format_plot_xaxis(ax=ax)
        offset.format_plot_yaxis(ax=ax)

        if add_cbar:
            label = f"Containment radius R{100 * fraction:.0f} ({containment.unit})"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        return ax

    def plot_psf_vs_rad(
        self, ax=None, offset=[0] * u.deg, energy_true=[0.1, 1, 10] * u.TeV, **kwargs
    ):
        """Plot PSF vs rad.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Axes to plot on.
        offset : `~astropy.coordinates.Angle`
            Offset in the field of view. Default offset = 0 deg
        energy_true : `~astropy.units.Quantity`
            True energy at which to plot the profile

        """
        from gammapy.datasets.map import RAD_AXIS_DEFAULT

        ax = plt.gca() if ax is None else ax

        try:
            rad = self.axes["rad"]
        except KeyError:
            rad = RAD_AXIS_DEFAULT

        for theta in offset:
            for energy in energy_true:
                psf_value = self.evaluate(
                    rad=rad.center, energy_true=energy, offset=theta
                )
                label = f"Offset: {theta:.1f}, Energy: {energy:.1f}"
                with quantity_support():
                    ax.plot(rad.center, psf_value, label=label, **kwargs)

        rad.format_plot_xaxis(ax=ax)

        ax.set_yscale("log")
        ax.set_ylabel(f"PSF ({ax.yaxis.units})")
        plt.legend()
        return ax

    def peek(self, figsize=(15, 5)):
        """Quick-look summary plots.

        Parameters
        ----------
        figsize : tuple
            Size of the figure.

        """
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)

        self.plot_containment_radius(fraction=0.68, ax=axes[0])
        self.plot_containment_radius(fraction=0.95, ax=axes[1])
        self.plot_containment_radius_vs_energy(ax=axes[2])
        plt.tight_layout()

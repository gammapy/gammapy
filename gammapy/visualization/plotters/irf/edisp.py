# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import Angle
from astropy.visualization import quantity_support
from matplotlib.colors import PowerNorm

from gammapy.visualization.plotters.core import BasePlotter
from gammapy.visualization.utils import add_colorbar
from .kernel import EDispKernelPlotter

__all__ = ["EDispPlotter"]


class EDispPlotter(BasePlotter):
    """Plotter for `~gammapy.irf.EnergyDispersion2D` objects.

    A plotter carries a local matplotlib configuration and exposes plotting
    methods that receive the `~gammapy.irf.EnergyDispersion2D` to plot as an
    argument. It keeps no reference to a data object.

    Parameters
    ----------
    rc_params : dict, optional
        Mapping of `matplotlib.rcParams` keys to values. Entries are validated
        when set and applied while plotting. Default is None.
    """



    def plot_migration(self, edisp, ax=None, offset=None, energy_true=None, **kwargs):
        """Plot energy dispersion for given offset and true energy.

        Parameters
        ----------
        edisp : `~gammapy.irf.EnergyDispersion2D`
            Energy dispersion to plot.
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes. Default is None.
        offset : `~astropy.coordinates.Angle`, optional
            Offset. Default is None.
        energy_true : `~astropy.units.Quantity`, optional
            True energy. Default is None.
        **kwargs : dict
            Keyword arguments forwarded to `~matplotlib.pyplot.plot`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Matplotlib axes.
        """
        with self._rc_context(), quantity_support():
            ax = plt.gca() if ax is None else ax

            if offset is None:
                offset = edisp._default_offset
            else:
                offset = np.atleast_1d(Angle(offset))

            if energy_true is None:
                energy_true = u.Quantity([0.1, 1, 10], "TeV")
            else:
                energy_true = np.atleast_1d(u.Quantity(energy_true))

            migra = edisp.axes["migra"]

            for ener in energy_true:
                for off in offset:
                    disp = edisp.evaluate(
                        offset=off, energy_true=ener, migra=migra.center
                    )
                    label = f"offset = {off:.1f}\nenergy = {ener:.1f}"
                    ax.plot(migra.center, disp, label=label, **kwargs)

            migra.format_plot_xaxis(ax=ax)
            ax.set_ylabel("Probability density")
            ax.legend(loc="upper left")
            return ax

    def plot_bias(
        self,
        edisp,
        ax=None,
        offset=None,
        add_cbar=False,
        axes_loc=None,
        kwargs_colorbar=None,
        **kwargs,
    ):
        """Plot migration as a function of true energy for a given offset.

        Parameters
        ----------
        edisp : `~gammapy.irf.EnergyDispersion2D`
            Energy dispersion to plot.
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes. Default is None.
        offset : `~astropy.coordinates.Angle`, optional
            Offset. Default is None.
        add_cbar : bool, optional
            Add a colorbar to the plot. Default is False.
        axes_loc : dict, optional
            Keyword arguments passed to `~mpl_toolkits.axes_grid1.axes_divider.AxesDivider.append_axes`.
        kwargs_colorbar : dict, optional
            Keyword arguments passed to `~matplotlib.pyplot.colorbar`.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.pcolormesh`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Matplotlib axes.
        """
        with self._rc_context(), quantity_support():
            kwargs.setdefault("cmap", "GnBu")
            kwargs.setdefault("norm", PowerNorm(gamma=0.5))

            kwargs_colorbar = kwargs_colorbar or {}

            ax = plt.gca() if ax is None else ax

            if offset is None:
                offset = edisp._default_offset

            energy_true = edisp.axes["energy_true"]
            migra = edisp.axes["migra"]

            z = edisp.evaluate(
                offset=offset,
                energy_true=energy_true.center.reshape(1, -1, 1),
                migra=migra.center.reshape(1, 1, -1),
            ).value[0]

            caxes = ax.pcolormesh(energy_true.edges, migra.edges, z.T, **kwargs)

            energy_true.format_plot_xaxis(ax=ax)
            migra.format_plot_yaxis(ax=ax)

            if add_cbar:
                label = "Probability density [A.U]."
                kwargs_colorbar.setdefault("label", label)
                add_colorbar(caxes, ax=ax, axes_loc=axes_loc, **kwargs_colorbar)

            return ax

    def peek(self, edisp, figsize=(15, 5)):
        """Quick-look summary plots.

        This method creates a figure with three subplots:

        * Bias plot : migration as a function of true energy for a given offset
        * Migration matrix plot : energy dispersion for given offset and true energy
        * Energy dispersion matrix plot : probability density function matrix to have
          ``energy`` as a function of ``energy_true``

        Parameters
        ----------
        edisp : `~gammapy.irf.EnergyDispersion2D`
            Energy dispersion to plot.
        figsize : tuple, optional
            Size of the resulting plot. Default is (15, 5).
        """
        with self._rc_context():
            _, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
            self.plot_bias(edisp, ax=axes[0])
            self.plot_migration(edisp, ax=axes[1])
            kernel = edisp.to_edisp_kernel(offset=edisp._default_offset[0])
            EDispKernelPlotter().plot_matrix(kernel, ax=axes[2])

            plt.tight_layout()

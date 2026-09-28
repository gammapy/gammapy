# Licensed under a 3-clause BSD style license - see LICENSE.rst
import matplotlib.pyplot as plt
from astropy.visualization import quantity_support
from matplotlib.colors import PowerNorm

from gammapy.maps.axes import UNIT_STRING_FORMAT
from gammapy.visualization.plotters.core import BasePlotter
from gammapy.visualization.utils import add_colorbar

__all__ = ["EDispKernelPlotter"]


class EDispKernelPlotter(BasePlotter):
    """Plotter for `~gammapy.irf.EDispKernel` objects.

    A plotter carries a local matplotlib configuration and exposes plotting
    methods that receive the `~gammapy.irf.EDispKernel` to plot as an
    argument. It keeps no reference to a data object.

    Parameters
    ----------
    rc_params : dict, optional
        Mapping of `matplotlib.rcParams` keys to values. Entries are validated
        when set and applied while plotting. Default is None.
    """



    def plot_matrix(
        self,
        kernel,
        ax=None,
        add_cbar=False,
        axes_loc=None,
        kwargs_colorbar=None,
        **kwargs,
    ):
        """Plot PDF matrix.

        Parameters
        ----------
        kernel : `~gammapy.irf.EDispKernel`
            Energy dispersion matrix to plot.
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes. Default is None.
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
            norm = PowerNorm(gamma=0.5, vmin=0, vmax=1)
            kwargs.setdefault("norm", norm)

            kwargs_colorbar = kwargs_colorbar or {}

            ax = plt.gca() if ax is None else ax

            energy_axis_true = kernel.axes["energy_true"]
            energy_axis = kernel.axes["energy"]

            caxes = ax.pcolormesh(
                energy_axis_true.edges, energy_axis.edges, kernel.data.T, **kwargs
            )

            if add_cbar:
                label = "Probability density (A.U.)"
                kwargs_colorbar.setdefault("label", label)
                add_colorbar(caxes, ax=ax, axes_loc=axes_loc, **kwargs_colorbar)

            energy_axis_true.format_plot_xaxis(ax=ax)
            energy_axis.format_plot_yaxis(ax=ax)
            return ax

    def plot_bias(self, kernel, ax=None, **kwargs):
        """Plot reconstruction bias.

        See `~gammapy.irf.EDispKernel.get_bias` method.

        Parameters
        ----------
        kernel : `~gammapy.irf.EDispKernel`
            Energy dispersion matrix to plot.
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes. Default is None.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Matplotlib axes.
        """
        with self._rc_context(), quantity_support():
            ax = plt.gca() if ax is None else ax

            energy = kernel.axes["energy_true"].center
            bias = kernel.get_bias(energy)

            ax.plot(energy, bias, **kwargs)

            ax.set_xlabel(
                f"$E_\\mathrm{{True}}$ [{ax.xaxis.units.to_string(UNIT_STRING_FORMAT)}]"
            )
            ax.set_ylabel(
                "($E_\\mathrm{{Reco}} - E_\\mathrm{{True}}) / E_\\mathrm{{True}}$"
            )
            ax.set_xscale("log")
            return ax

    def peek(self, kernel, figsize=(15, 5)):
        """Quick-look summary plots.

        This method creates a figure with two subplots:

        * Bias plot : reconstruction bias as a function of true energy
        * Energy dispersion matrix plot : probability density function matrix to have
          ``energy`` as a function of ``energy_true``

        Parameters
        ----------
        kernel : `~gammapy.irf.EDispKernel`
            Energy dispersion matrix to plot.
        figsize : tuple, optional
            Size of the figure. Default is (15, 5).
        """
        with self._rc_context():
            _, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
            self.plot_bias(kernel, ax=axes[0])
            self.plot_matrix(kernel, ax=axes[1])
            plt.tight_layout()

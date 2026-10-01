# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from gammapy.maps.axes import UNIT_STRING_FORMAT
from gammapy.visualization.plotters.core import BasePlotter

__all__ = ["RadMaxPlotter"]


class RadMaxPlotter(BasePlotter):
    """Plotter for `~gammapy.irf.RadMax2D` objects.

    A plotter carries a local matplotlib configuration and exposes plotting
    methods that receive the `~gammapy.irf.RadMax2D` to plot as an
    argument. It keeps no reference to a data object.

    Parameters
    ----------
    rc_params : dict, optional
        Mapping of `matplotlib.rcParams` keys to values. Entries are
        validated when set and applied while plotting. Default is None.
    """

    def plot(self, rad_max, ax=None, **kwargs):
          """Plot radial maximum values against energy.
           
          A separate line is drawn for each offset bin of the input
          ``rad_max`` object.
           
          Parameters
          ----------
          rad_max : `~gammapy.irf.RadMax2D`
              Radial maximum table to plot.
          ax : `~matplotlib.axes.Axes`, optional
              Matplotlib axes. If not provided, the axes passed at
              initialization or the current axes are used.
          **kwargs : dict
              Keyword arguments forwarded to
              `~matplotlib.axes.Axes.plot`.
           
          Returns
          -------
          ax : `~matplotlib.axes.Axes`
              Matplotlib axes containing the plot.
          """    
        ax = plt.gca() if ax is None else ax

        energy_axis = rad_max.axes["energy"]
        offset_axis = rad_max.axes["offset"]

        with quantity_support():
            for value in offset_axis.center:
                rad_max_val = rad_max.evaluate(offset=value)
                line_kwargs = kwargs.copy()
                line_kwargs.setdefault("label", f"Offset {value:.2f}")
                ax.plot(energy_axis.center, rad_max_val, **line_kwargs)

        energy_axis.format_plot_xaxis(ax=ax)
        ax.set_ylim(0 * rad_max.unit, None)
        ax.legend(loc="best")
        ax.set_ylabel(f"Rad max. [{ax.yaxis.units.to_string(UNIT_STRING_FORMAT)}]")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        return ax

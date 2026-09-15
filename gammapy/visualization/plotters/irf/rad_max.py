# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from gammapy.maps.axes import UNIT_STRING_FORMAT
from gammapy.visualization.plotters.core import BasePlotter

__all__ = [
    "RadMaxPlotter",
]


class RadMaxPlotter(BasePlotter):
    def plot(self, rad_max, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

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

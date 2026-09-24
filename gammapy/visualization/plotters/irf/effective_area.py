# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import quantity_support
from gammapy.visualization import add_colorbar
from gammapy.maps.axes import UNIT_STRING_FORMAT
from ..core import BasePlotter


class EffectiveAreaPlotter(BasePlotter):
    def plot(
        self,
        aeff,
        ax=None,
        add_cbar=True,
        axes_loc=None,
        kwargs_colorbar=None,
        **kwargs,
    ):
        with self._rc_context():

            ax = plt.gca() if ax is None else ax

            energy = aeff.axes["energy_true"]
            offset = aeff.axes["offset"]
            aeff_val = aeff.evaluate(
                offset=offset.center, energy_true=energy.center[:, np.newaxis]
            )

            vmin, vmax = np.nanmin(aeff_val.value), np.nanmax(aeff_val.value)

            kwargs.setdefault("cmap", "GnBu")
            kwargs.setdefault("edgecolors", "face")
            kwargs.setdefault("vmin", vmin)
            kwargs.setdefault("vmax", vmax)

            kwargs_colorbar = kwargs_colorbar or {}

            with quantity_support():
                caxes = ax.pcolormesh(
                    energy.edges, offset.edges, aeff_val.value.T, **kwargs
                )

            energy.format_plot_xaxis(ax=ax)
            offset.format_plot_yaxis(ax=ax)

            if add_cbar:
                label = f"Effective Area [{aeff.unit.to_string(UNIT_STRING_FORMAT)}]"
                kwargs_colorbar.setdefault("label", label)
                add_colorbar(caxes, ax=ax, axes_loc=axes_loc, **kwargs_colorbar)

            return ax

    def plot_offset_dependence(self, aeff, ax=None, energy=None, **kwargs):
        with self._rc_context():

            ax = plt.gca() if ax is None else ax

            if energy is None:
                energy_axis = aeff.axes["energy_true"]
                e_min, e_max = energy_axis.center[[0, -1]]
                energy = np.geomspace(e_min, e_max, 4)

            offset_axis = aeff.axes["offset"]

            with quantity_support():
                for ee in energy:
                    area = aeff.evaluate(offset=offset_axis.center, energy_true=ee)
                    max_area = np.nanmax(area)
                    if max_area == 0:
                        continue
                    area /= max_area
                    if np.isnan(area).all():
                        continue
                    label = f"energy = {ee:.1f}"
                    ax.plot(offset_axis.center, area, label=label, **kwargs)

            offset_axis.format_plot_xaxis(ax=ax)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Relative Effective Area")
            ax.legend(loc="best")
            return ax

    def plot_energy_dependence(self, aeff, ax=None, offset=None, **kwargs):
        with self._rc_context():
            ax = plt.gca() if ax is None else ax

            if offset is None:
                off_min, off_max = aeff.axes["offset"].bounds
                offset = np.linspace(off_min, off_max, 4)

            energy_axis = aeff.axes["energy_true"]
            with quantity_support():
                for off in offset:
                    area = aeff.evaluate(offset=off, energy_true=energy_axis.center)
                    label = kwargs.pop("label", f"offset = {off:.1f}")
                    ax.plot(energy_axis.center, area, label=label, **kwargs)

            energy_axis.format_plot_xaxis(ax=ax)
            ax.set_ylabel(
                f"Effective Area [{ax.yaxis.units.to_string(UNIT_STRING_FORMAT)}]"
            )
            ax.legend()
            return ax

    def peek(self, aeff, figsize=(15, 5)):
        with self._rc_context():
            ncols = 2 if aeff.is_pointlike else 3
            _, axes = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)
            self.plot(aeff, ax=axes[ncols - 1])
            self.plot_energy_dependence(aeff, ax=axes[0])
            if aeff.is_pointlike is False:
                self.plot_offset_dependence(aeff, ax=axes[1])
            plt.tight_layout()

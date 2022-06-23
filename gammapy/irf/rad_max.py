# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
from .core import IRF

__all__ = [
    "RadMax2D",
]


class RadMax2D(IRF):
    """2D Rad Max table.

    This is not directly a IRF component but is needed as additional information
    for point-like IRF components when an energy or field of view
    dependent directional cut has been applied.

    Data format specification: :ref:`gadf:rad_max_2d`

    Parameters
    ----------
    energy_axis : `MapAxis`
        Reconstructed energy axis
    offset_axis : `MapAxis`
        Field of view offset axis.
    data : `~astropy.units.Quantity`
        Applied directional cut
    meta : dict
        Meta data
    """

    tag = "rad_max_2d"
    required_axes = ["energy", "offset"]
    default_unit = u.deg

    @classmethod
    def from_irf(cls, irf):
        """Create a RadMax2D instance from another IRF component.

        This reads the RAD_MAX metadata keyword from the irf and creates
        a RadMax2D with a single bin in energy and offset using the
        ranges from the input irf.

        Parameters
        ----------
        irf: `~gammapy.irf.EffectiveAreaTable2D` or `~gammapy.irf.EnergyDispersion2D`
            IRF instance from which to read the RAD_MAX and limit information

        Returns
        -------
        rad_max: `RadMax2D`
            `RadMax2D` object with a single bin corresponding to the fixed
            RAD_MAX cut.

        Notes
        -----
        This assumes the true energy axis limits are also valid for the
        reco energy limits.
        """
        if not irf.is_pointlike:
            raise ValueError("RadMax2D.from_irf requires a point-like irf")

        if "RAD_MAX" not in irf.meta:
            raise ValueError("Irf does not contain RAD_MAX keyword")

        rad_max_value = irf.meta["RAD_MAX"]
        if not isinstance(rad_max_value, float):
            raise ValueError(
                f"RAD_MAX must be a float, got '{type(rad_max_value)}' instead"
            )

        energy_axis = irf.axes["energy_true"].copy(name="energy").squash()
        offset_axis = irf.axes["offset"].squash()

        return cls(
            data=rad_max_value,
            axes=[energy_axis, offset_axis],
            unit="deg",
            interp_kwargs={"method": "nearest", "fill_value": None},
        )

    def plot_rad_max_vs_energy(self, ax=None, **kwargs):
        """Plot rad max value against energy.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Axes to plot on.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.pcolormesh`

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
             Axes to plot on.
        """
        if ax is None:
            ax = plt.gca()

        energy_axis = self.axes["energy"]
        offset_axis = self.axes["offset"]

        with quantity_support():
            for value in offset_axis.center:
                rad_max = self.evaluate(offset=value)
                label = f"Offset {value:.2f}"
                kwargs.setdefault("label", label)
                ax.plot(energy_axis.center, rad_max, **kwargs)

        energy_axis.format_plot_xaxis(ax=ax)
        ax.set_ylim(0 * u.deg, None)
        ax.legend(loc="best")
        ax.set_ylabel(f"Rad max. [{ax.yaxis.units}]")
        ax.yaxis.set_major_formatter("{x:.1f}")
        return ax

    @property
    def is_fixed_rad_max(self):
        """Returns True if rad_max axes are flat."""
        return self.axes.is_flat

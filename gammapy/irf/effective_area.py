# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
from gammapy.maps import MapAxes, MapAxis
from .core import IRF

__all__ = ["EffectiveAreaTable2D"]


class EffectiveAreaTable2D(IRF):
    """2D effective area table.

    Data format specification: :ref:`gadf:aeff_2d`

    Parameters
    ----------
    energy_axis_true : `MapAxis`
        True energy axis
    offset_axis : `MapAxis`
        Field of view offset axis.
    data : `~astropy.units.Quantity`
        Effective area
    meta : dict
        Meta data

    Examples
    --------
    Here's an example you can use to learn about this class:

    >>> from gammapy.irf import EffectiveAreaTable2D
    >>> filename = '$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits'
    >>> aeff = EffectiveAreaTable2D.read(filename, hdu='EFFECTIVE AREA')
    >>> print(aeff)
    EffectiveAreaTable2D
    --------------------
    <BLANKLINE>
      axes  : ['energy_true', 'offset']
      shape : (42, 6)
      ndim  : 2
      unit  : m2
      dtype : >f4
    <BLANKLINE>

    Here's another one, created from scratch, without reading a file:

    >>> from gammapy.irf import EffectiveAreaTable2D
    >>> from gammapy.maps import MapAxis
    >>> energy_axis_true = MapAxis.from_energy_bounds(
            "0.1 TeV", "100 TeV", nbin=30, name="energy_true"
        )
    >>> offset_axis = MapAxis.from_bounds(0, 5, nbin=4, name="offset")
    >>> aeff = EffectiveAreaTable2D(axes=[energy_axis_true, offset_axis], data=1e10, unit="cm2")
    >>> print(aeff)
    EffectiveAreaTable2D
    --------------------
    <BLANKLINE>
      axes  : ['energy_true', 'offset']
      shape : (30, 4)
      ndim  : 2
      unit  : cm2
      dtype : float64
    <BLANKLINE>

    """

    tag = "aeff_2d"
    required_axes = ["energy_true", "offset"]
    default_unit = u.m**2

    def plot_energy_dependence(self, ax=None, offset=None, **kwargs):
        """Plot effective area versus energy for a given offset.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        offset : `~astropy.coordinates.Angle`
            Offset
        kwargs : dict
            Forwarded tp plt.plot()

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        ax = plt.gca() if ax is None else ax

        if offset is None:
            off_min, off_max = self.axes["offset"].bounds
            offset = np.linspace(off_min, off_max, 4)

        energy_axis = self.axes["energy_true"]

        for off in offset:
            area = self.evaluate(offset=off, energy_true=energy_axis.center)
            label = kwargs.pop("label", f"offset = {off:.1f}")
            with quantity_support():
                ax.plot(energy_axis.center, area, label=label, **kwargs)

        energy_axis.format_plot_xaxis(ax=ax)
        ax.set_ylabel(f"Effective Area [{ax.yaxis.units}]")
        ax.legend()
        return ax

    def plot_offset_dependence(self, ax=None, energy=None, **kwargs):
        """Plot effective area versus offset for a given energy.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        energy : `~astropy.units.Quantity`
            Energy
        **kwargs : dict
            Keyword argument passed to `~matplotlib.pyplot.plot`

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        ax = plt.gca() if ax is None else ax

        if energy is None:
            energy_axis = self.axes["energy_true"]
            e_min, e_max = energy_axis.center[[0, -1]]
            energy = np.geomspace(e_min, e_max, 4)

        offset_axis = self.axes["offset"]

        for ee in energy:
            area = self.evaluate(offset=offset_axis.center, energy_true=ee)
            area /= np.nanmax(area)
            if np.isnan(area).all():
                continue
            label = f"energy = {ee:.1f}"
            with quantity_support():
                ax.plot(offset_axis.center, area, label=label, **kwargs)

        offset_axis.format_plot_xaxis(ax=ax)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Relative Effective Area")
        ax.legend(loc="best")
        return ax

    def plot(self, ax=None, add_cbar=True, **kwargs):
        """Plot effective area image."""
        ax = plt.gca() if ax is None else ax

        energy = self.axes["energy_true"]
        offset = self.axes["offset"]
        aeff = self.evaluate(
            offset=offset.center, energy_true=energy.center[:, np.newaxis]
        )

        vmin, vmax = np.nanmin(aeff.value), np.nanmax(aeff.value)

        kwargs.setdefault("cmap", "GnBu")
        kwargs.setdefault("edgecolors", "face")
        kwargs.setdefault("vmin", vmin)
        kwargs.setdefault("vmax", vmax)

        with quantity_support():
            caxes = ax.pcolormesh(energy.edges, offset.edges, aeff.value.T, **kwargs)

        energy.format_plot_xaxis(ax=ax)
        offset.format_plot_yaxis(ax=ax)

        if add_cbar:
            label = f"Effective Area [{aeff.unit}]"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        return ax

    def peek(self, figsize=(15, 5)):
        """Quick-look summary plots.

        Parameters
        ----------
        figsize : tuple
            Size of the figure.

        """
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        self.plot(ax=axes[2])
        self.plot_energy_dependence(ax=axes[0])
        self.plot_offset_dependence(ax=axes[1])
        plt.tight_layout()

    @classmethod
    def from_parametrization(cls, energy_axis_true=None, instrument="HESS"):
        r"""Create parametrized effective area.

        Parametrizations of the effective areas of different Cherenkov
        telescopes taken from Appendix B of Abramowski et al. (2010), see
        https://ui.adsabs.harvard.edu/abs/2010MNRAS.402.1342A .

        .. math::
            A_{eff}(E) = g_1 \left(\frac{E}{\mathrm{MeV}}\right)^{-g_2}\exp{\left(-\frac{g_3}{E}\right)}  # noqa: E501

        This method does not model the offset dependence of the effective area,
        but just assumes that it is constant.

        Parameters
        ----------
        energy_axis_true : `MapAxis`
            Energy binning, analytic function is evaluated at log centers
        instrument : {'HESS', 'HESS2', 'CTA'}
            Instrument name

        Returns
        -------
        aeff : `EffectiveAreaTable2D`
            Effective area table
        """
        # Put the parameters g in a dictionary.
        # Units: g1 (cm^2), g2 (), g3 (MeV)
        pars = {
            "HESS": [6.85e9, 0.0891, 5e5],
            "HESS2": [2.05e9, 0.0891, 1e5],
            "CTA": [1.71e11, 0.0891, 1e5],
        }

        if instrument not in pars.keys():
            ss = f"Unknown instrument: {instrument}\n"
            ss += f"Valid instruments: {list(pars.keys())}"
            raise ValueError(ss)

        if energy_axis_true is None:
            energy_axis_true = MapAxis.from_energy_bounds(
                "2 GeV", "200 TeV", nbin=20, per_decade=True, name="energy_true"
            )

        g1, g2, g3 = pars[instrument]

        offset_axis = MapAxis.from_edges([0.0, 5.0] * u.deg, name="offset")
        axes = MapAxes([energy_axis_true, offset_axis])
        coords = axes.get_coord()

        energy, offset = coords["energy_true"].to_value("MeV"), coords["offset"]
        data = np.ones_like(offset.value) * g1 * energy ** (-g2) * np.exp(-g3 / energy)

        # TODO: fake offset dependence?
        meta = {"TELESCOP": instrument}
        return cls(axes=axes, data=data, unit="cm2", meta=meta)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
import astropy.units as u
from astropy.coordinates import angular_separation
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from gammapy.maps import MapAxes, MapAxis
from gammapy.maps.axes import UNIT_STRING_FORMAT
from gammapy.utils.compat import COPY_IF_NEEDED
from gammapy.visualization.utils import add_colorbar
from .core import IRF
from .io import gadf_is_pointlike

__all__ = ["Background3D", "Background2D", "BackgroundIRF"]

log = logging.getLogger(__name__)


class BackgroundIRF(IRF):
    """Background IRF base class.

    Parameters
    ----------
    axes : list of `MapAxis` or `MapAxes` object
    data : `~np.ndarray`
        Data array.
    unit : str or `~astropy.units.Unit`
        Data unit usually ``s^-1 MeV^-1 sr^-1``.
    meta : dict
        Metadata dictionary.

    """

    default_interp_kwargs = dict(bounds_error=False, fill_value=0.0, values_scale="log")
    """Default Interpolation kwargs to extrapolate."""

    @classmethod
    def from_table(cls, table, format="gadf-dl3"):
        """Read from `~astropy.table.Table`.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table with background data.
        format : {"gadf-dl3"}
            Format specification. Default is "gadf-dl3".

        Returns
        -------
        bkg : `Background2D` or `Background2D`
            Background IRF class.
        """
        # TODO: some of the existing background files have missing HDUCLAS keywords
        #  which are required to define the correct Gammapy axis names

        if "HDUCLAS2" not in table.meta:
            log.warning("Missing 'HDUCLAS2' keyword assuming 'BKG'")
            table = table.copy()
            table.meta["HDUCLAS2"] = "BKG"

        axes = MapAxes.from_table(table, format=format)[cls.required_axes]

        # TODO: spec says key should be "BKG", but there are files around
        #  (e.g. CTA 1DC) that use "BGD". For now we support both
        if "BKG" in table.colnames:
            bkg_name = "BKG"
        elif "BGD" in table.colnames:
            bkg_name = "BGD"
        else:
            raise ValueError("Invalid column names. Need 'BKG' or 'BGD'.")

        data = table[bkg_name].quantity[0].T

        if data.unit == "" or isinstance(data.unit, u.UnrecognizedUnit):
            data = u.Quantity(data.value, "s-1 MeV-1 sr-1", copy=COPY_IF_NEEDED)
            log.warning(
                "Invalid unit found in background table! Assuming (s-1 MeV-1 sr-1)"
            )

        # TODO: The present HESS and CTA background fits files
        #  have a reverse order (lon, lat, E) than recommended in GADF(E, lat, lon)
        #  For now, we support both.

        if axes.shape == axes.shape[::-1]:
            log.error("Ambiguous axes order in Background fits files!")

        if np.shape(data) != axes.shape:
            log.debug("Transposing background table on read")
            data = data.transpose()

        return cls(
            axes=axes,
            data=data.value,
            meta=table.meta,
            unit=data.unit,
            is_pointlike=gadf_is_pointlike(table.meta),
            fov_alignment=table.meta.get("FOVALIGN", "RADEC"),
        )


class Background3D(BackgroundIRF):
    """Background 3D.

    Data format specification: :ref:`gadf:bkg_3d`.

    Parameters
    ----------
    axes : list of `MapAxis` or `MapAxes` object
        Required data axes: ["energy", "fov_lon", "fov_lat"] in the given order.
    data : `~np.ndarray`
        Data array.
    unit : str or `~astropy.units.Unit`
        Data unit usually ``s^-1 MeV^-1 sr^-1``.
    fov_alignment : `~gammapy.irf.FoVAlignment`
        The orientation of the field of view coordinate system.
    meta : dict
        Metadata dictionary.

    Examples
    --------
    Here's an example you can use to learn about this class:

    >>> from gammapy.irf import Background3D
    >>> filename = '$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits'
    >>> bkg_3d = Background3D.read(filename, hdu='BACKGROUND')
    >>> print(bkg_3d)
    Background3D
    ------------
    <BLANKLINE>
      axes  : ['energy', 'fov_lon', 'fov_lat']
      shape : (21, 36, 36)
      ndim  : 3
      unit  : 1 / (MeV s sr)
      dtype : >f4
    <BLANKLINE>

    """

    tag = "bkg_3d"
    required_axes = ["energy", "fov_lon", "fov_lat"]
    default_unit = u.s**-1 * u.MeV**-1 * u.sr**-1

    def to_2d(self):
        """Convert to `Background2D`.

        This takes the values at Y = 0 and X >= 0.
        """
        # TODO: this is incorrect as it misses the Jacobian?

        idx_lon = self.axes["fov_lon"].coord_to_idx(0 * u.deg)[0]
        idx_lat = self.axes["fov_lat"].coord_to_idx(0 * u.deg)[0]
        data = self.quantity[:, idx_lon:, idx_lat].copy()

        offset = self.axes["fov_lon"].edges[idx_lon:]
        offset_axis = MapAxis.from_edges(offset, name="offset")

        return Background2D(
            axes=[self.axes["energy"], offset_axis], data=data.value, unit=data.unit
        )

    def peek(self, figsize=(10, 8)):
        """Quick-look summary plots.

        Parameters
        ----------
        figsize : tuple, optional
            Size of the figure. Default is (10, 8).

        """
        return self.to_2d().peek(figsize)

    def plot_at_energy(
        self,
        energy=1 * u.TeV,
        add_cbar=True,
        ncols=3,
        figsize=None,
        axes_loc=None,
        kwargs_colorbar=None,
        **kwargs,
    ):
        """Plot the background rate in FoV coordinates at a given energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`, optional
            List of energies. Default is 1 TeV.
        add_cbar : bool, optional
            Add color bar. Default is True.
        ncols : int, optional
            Number of columns to plot. Default is 3.
        figsize : tuple, optional
            Figure size. Default is None.
        axes_loc : dict, optional
            Keyword arguments passed to `~mpl_toolkits.axes_grid1.axes_divider.AxesDivider.append_axes`.
        kwargs_colorbar : dict, optional
            Keyword arguments passed to `~matplotlib.pyplot.colorbar`.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.pcolormesh`.
        """
        kwargs_colorbar = kwargs_colorbar or {}

        n = len(energy)
        cols = min(ncols, n)
        rows = 1 + (n - 1) // cols
        width = 12
        cfraction = 0.0
        if add_cbar:
            cfraction = 0.15
        if figsize is None:
            figsize = (width, rows * width // (cols * (1 + cfraction)))

        fig, axes = plt.subplots(
            ncols=cols,
            nrows=rows,
            figsize=figsize,
            gridspec_kw={"hspace": 0.2, "wspace": 0.3},
        )

        x = self.axes["fov_lat"].edges
        y = self.axes["fov_lon"].edges
        X, Y = np.meshgrid(x, y)

        for i, ee in enumerate(energy):
            if len(energy) == 1:
                ax = axes
            else:
                ax = axes.flat[i]
            bkg = self.evaluate(energy=ee)
            bkg_unit = bkg.unit
            bkg = bkg.value
            with quantity_support():
                caxes = ax.pcolormesh(X, Y, bkg.squeeze(), **kwargs)

            self.axes["fov_lat"].format_plot_xaxis(ax)
            self.axes["fov_lon"].format_plot_yaxis(ax)
            ax.set_title(str(ee))

            if add_cbar:
                label = f"Background [{bkg_unit.to_string(UNIT_STRING_FORMAT)}]"
                kwargs_colorbar.setdefault("label", label)
                cbar = add_colorbar(caxes, ax=ax, axes_loc=axes_loc, **kwargs_colorbar)
                cbar.formatter.set_powerlimits((0, 0))

            row, col = np.unravel_index(i, shape=(rows, cols))
            if col > 0:
                ax.set_ylabel("")
            if row < rows - 1:
                ax.set_xlabel("")
            ax.set_aspect("equal", "box")


class Background2D(BackgroundIRF):
    """Background 2D.

    Data format specification: :ref:`gadf:bkg_2d`

    Parameters
    ----------
    axes : list of `MapAxis` or `MapAxes` object
        Required data axes: ["energy", "offset"] in the given order.
    data : `~np.ndarray`
        Data array.
    unit : str or `~astropy.units.Unit`
        Data unit usually ``s^-1 MeV^-1 sr^-1``.
    meta : dict
        Metadata dictionary.
    """

    tag = "bkg_2d"
    required_axes = ["energy", "offset"]
    default_unit = u.s**-1 * u.MeV**-1 * u.sr**-1

    def to_3d(self):
        """Convert to Background3D."""
        offsets = self.axes["offset"].edges
        edges_neg = np.negative(offsets)[::-1]
        edges_neg = edges_neg[edges_neg <= 0]
        edges = np.concatenate((edges_neg, offsets[offsets > 0]))
        fov_lat = MapAxis.from_edges(edges=edges, name="fov_lat")
        fov_lon = MapAxis.from_edges(edges=edges, name="fov_lon")

        axes = MapAxes([self.axes["energy"], fov_lon, fov_lat])
        coords = axes.get_coord()
        offset = angular_separation(
            0 * u.rad, 0 * u.rad, coords["fov_lon"], coords["fov_lat"]
        )
        data = self.evaluate(offset=offset, energy=coords["energy"])

        return Background3D(
            axes=axes,
            data=data,
        )

    def plot_at_energy(
        self, energy=1 * u.TeV, add_cbar=True, ncols=3, figsize=None, **kwargs
    ):
        """Plot the background rate in FoV coordinates at a given energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`, optional
            List of energy. Default is 1 TeV.
        add_cbar : bool, optional
            Add color bar. Default is True.
        ncols : int, optional
            Number of columns to plot. Default is 3.
        figsize : tuple, optional
            Figure size. Default is None.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.pcolormesh`.
        """
        bkg_3d = self.to_3d()
        bkg_3d.plot_at_energy(
            energy=energy, add_cbar=add_cbar, ncols=ncols, figsize=figsize, **kwargs
        )

    def plot(
        self, ax=None, add_cbar=True, axes_loc=None, kwargs_colorbar=None, **kwargs
    ):
        """Plot energy offset dependence of the background model.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes. Default is None.
        add_cbar : bool, optional
            Add a colorbar to the plot. Default is True.
        axes_loc : dict, optional
            Keyword arguments passed to `~mpl_toolkits.axes_grid1.axes_divider.AxesDivider.append_axes`.
        kwargs_colorbar : dict, optional
            Keyword arguments passed to `~matplotlib.pyplot.colorbar`.
        kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.pcolormesh`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Matplotlib axes.
        """
        ax = plt.gca() if ax is None else ax

        energy_axis, offset_axis = self.axes["energy"], self.axes["offset"]
        data = self.quantity.value

        kwargs.setdefault("cmap", "GnBu")
        kwargs.setdefault("edgecolors", "face")
        kwargs.setdefault("norm", LogNorm())

        kwargs_colorbar = kwargs_colorbar or {}

        with quantity_support():
            caxes = ax.pcolormesh(
                energy_axis.edges, offset_axis.edges, data.T, **kwargs
            )

        energy_axis.format_plot_xaxis(ax=ax)
        offset_axis.format_plot_yaxis(ax=ax)

        if add_cbar:
            label = (
                f"Background rate [{self.quantity.unit.to_string(UNIT_STRING_FORMAT)}]"
            )
            kwargs_colorbar.setdefault("label", label)
            add_colorbar(caxes, ax=ax, axes_loc=axes_loc, **kwargs_colorbar)

    def plot_offset_dependence(self, ax=None, energy=None, **kwargs):
        """Plot background rate versus offset for a given energy.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes. Default is None.
        energy : `~astropy.units.Quantity`, optional
            Energy. Default is None.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Matplotlib axes.
        """
        ax = plt.gca() if ax is None else ax

        if energy is None:
            energy_axis = self.axes["energy"]
            e_min, e_max = np.log10(energy_axis.center.value[[0, -1]])
            energy = np.logspace(e_min, e_max, 4) * energy_axis.unit

        offset_axis = self.axes["offset"]

        for ee in energy:
            bkg = self.evaluate(offset=offset_axis.center, energy=ee)
            if np.isnan(bkg).all():
                continue
            label = f"energy = {ee:.1f}"
            with quantity_support():
                ax.plot(offset_axis.center, bkg, label=label, **kwargs)

        offset_axis.format_plot_xaxis(ax=ax)
        ax.set_ylabel(
            f"Background rate [{ax.yaxis.units.to_string(UNIT_STRING_FORMAT)}]"
        )
        ax.set_yscale("log")
        ax.legend(loc="upper right")
        return ax

    def plot_energy_dependence(self, ax=None, offset=None, **kwargs):
        """Plot background rate versus energy for a given offset.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes. Default is None.
        offset : `~astropy.coordinates.Angle`, optional
            Offset. Default is None.
        kwargs : dict
            Forwarded to plt.plot().

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Matplotlib axes.
        """
        ax = plt.gca() if ax is None else ax

        if offset is None:
            offset_axis = self.axes["offset"]
            off_min, off_max = offset_axis.center.value[[0, -1]]
            offset = np.linspace(off_min, off_max, 4) * offset_axis.unit

        energy_axis = self.axes["energy"]

        for off in offset:
            bkg = self.evaluate(offset=off, energy=energy_axis.center)
            label = f"offset = {off:.2f}"
            with quantity_support():
                ax.plot(energy_axis.center, bkg, label=label, **kwargs)

        energy_axis.format_plot_xaxis(ax=ax)
        ax.set_yscale("log")
        ax.set_ylabel(
            f"Background rate [{ax.yaxis.units.to_string(UNIT_STRING_FORMAT)}]"
        )
        ax.legend(loc="best")
        return ax

    def plot_spectrum(self, ax=None, **kwargs):
        """Plot angle integrated background rate versus energy.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes. Default is None.
        **kwargs : dict
            Keyword arguments forwarded to `~matplotib.pyplot.plot`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Matplotlib axes.
        """
        ax = plt.gca() if ax is None else ax

        offset_axis = self.axes["offset"]
        energy_axis = self.axes["energy"]

        bkg = self.integral(offset=offset_axis.bounds[1], axis_name="offset")
        bkg = bkg.to(u.Unit("s-1") / energy_axis.unit)

        with quantity_support():
            ax.plot(energy_axis.center, bkg, label="integrated spectrum", **kwargs)

        energy_axis.format_plot_xaxis(ax=ax)
        ax.set_yscale("log")
        ax.set_ylabel(
            f"Background rate [{ax.yaxis.units.to_string(UNIT_STRING_FORMAT)}]"
        )
        ax.legend(loc="best")
        return ax

    def peek(self, figsize=(10, 8)):
        """Quick-look summary plots."""
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        self.plot(ax=axes[1][1])
        self.plot_offset_dependence(ax=axes[0][0])
        self.plot_energy_dependence(ax=axes[1][0])
        self.plot_spectrum(ax=axes[0][1])
        plt.tight_layout()

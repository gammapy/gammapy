# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
import astropy.units as u
from gammapy.maps import MapAxes, MapAxis
from gammapy.utils.integrate import trapz_loglog
from gammapy.utils.interpolation import interpolate_invalid_data_3d
from .core import IRF

__all__ = ["Background3D", "Background2D"]

log = logging.getLogger(__name__)


class BackgroundIRF(IRF):
    """Background IRF base class"""

    default_interp_kwargs = dict(
        bounds_error=False, fill_value=None, values_scale="log"
    )
    """Default Interpolation kwargs to extrapolate."""

    @classmethod
    def from_table(cls, table, format="gadf-dl3"):
        """Read from `~astropy.table.Table`.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table with background data
        format : {"gadf-dl3"}
            Format specification

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
            data = u.Quantity(data.value, "s-1 MeV-1 sr-1", copy=False)
            log.warning(
                "Invalid unit found in background table! Assuming (s-1 MeV-1 sr-1)"
            )

        # TODO: The present HESS and CTA background fits files
        #  have a reverse order (lon, lat, E) than recommened in GADF(E, lat, lon)
        #  For now, we suport both.

        if axes.shape == axes.shape[::-1]:
            log.error("Ambiguous axes order in Background fits files!")

        if np.shape(data) != axes.shape:
            log.debug("Transposing background table on read")
            data = data.transpose()

        return cls(axes=axes, data=data.value, meta=table.meta, unit=data.unit)


class Background3D(BackgroundIRF):
    """Background 3D.

    Data format specification: :ref:`gadf:bkg_3d`

    Parameters
    ----------
    axes : list of `MapAxis` or `MapAxes` object
        Required data axes: ["energy", "fov_lon", "fov_lat"] in the given order.
    data : `~np.ndarray`
        Data array.
    unit : str or `~astropy.units.Unit`
        Data unit usuually ``s^-1 MeV^-1 sr^-1``
    meta : dict
        Meta data
    interp_missing_values: bool
        Interpolate missing values in background 3d map.
        Default is True, have to be set to True for CTA IRF.

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

    def __init__(self, axes, data=0, unit="", meta=None, interp_missing_values=False):
        super().__init__(axes, data, unit, meta)
        if interp_missing_values:
            self.interp_missing_values()

    def interp_missing_values(self):
        """Interpolate missing values in bakcground 3d"""
        if isinstance(self.axes, list):
            axis = [a for a in self.axes if a.name == "energy"][0]
        else:
            axis = self.axes["energy"]
        log_energy = np.log(axis.center.value)
        log_data = np.ma.masked_invalid(np.log(self.data))
        log_data = interpolate_invalid_data_3d(log_data, log_energy)
        self.data = np.exp(log_data.data)

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
        return self.to_2d().peek(figsize)


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
        Data unit usually ``s^-1 MeV^-1 sr^-1``
    meta : dict
        Meta data
    """

    tag = "bkg_2d"
    required_axes = ["energy", "offset"]
    default_interp_kwargs = dict(bounds_error=False, fill_value=None)
    """Default Interpolation kwargs."""

    def plot(self, ax=None, add_cbar=True, **kwargs):
        """Plot energy offset dependence of the background model.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        ax = plt.gca() if ax is None else ax

        x = self.axes["energy"].edges.to_value("TeV")
        y = self.axes["offset"].edges.to_value("deg")
        z = self.quantity.T.value

        kwargs.setdefault("cmap", "GnBu")
        kwargs.setdefault("edgecolors", "face")

        caxes = ax.pcolormesh(x, y, z, norm=LogNorm(), **kwargs)
        ax.set_xscale("log")
        ax.set_ylabel(f"Offset (deg)")
        ax.set_xlabel(f"Energy (TeV)")

        xmin, xmax = x.min(), x.max()
        ax.set_xlim(xmin, xmax)

        if add_cbar:
            label = f"Background rate ({self.unit})"
            ax.figure.colorbar(caxes, ax=ax, label=label)

    def plot_offset_dependence(self, ax=None, energy=None, **kwargs):
        """Plot background rate versus offset for a given energy.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        energy : `~astropy.units.Quantity`
            Energy

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        if energy is None:
            energy_axis = self.axes["energy"]
            e_min, e_max = np.log10(energy_axis.center.value[[0, -1]])
            energy = np.logspace(e_min, e_max, 4) * energy_axis.unit

        offset = self.axes["offset"].center

        for ee in energy:
            bkg = self.evaluate(offset=offset, energy=ee)
            if np.isnan(bkg).all():
                continue
            label = f"energy = {ee:.1f}"
            ax.plot(offset, bkg.value, label=label, **kwargs)

        ax.set_xlabel(f"Offset ({self.axes['offset'].unit})")
        ax.set_ylabel(f"Background rate ({self.unit})")
        ax.set_yscale("log")
        ax.legend(loc="upper right")
        return ax

    def plot_energy_dependence(self, ax=None, offset=None, **kwargs):
        """Plot background rate versus energy for a given offset.

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
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        if offset is None:
            offset_axis = self.axes["offset"]
            off_min, off_max = offset_axis.center.value[[0, -1]]
            offset = np.linspace(off_min, off_max, 4) * offset_axis.unit

        energy = self.axes["energy"].center

        for off in offset:
            bkg = self.evaluate(offset=off, energy=energy)
            label = f"offset = {off:.2f}"
            ax.plot(energy, bkg.value, label=label, **kwargs)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"Energy [{energy.unit}]")
        ax.set_ylabel(f"Background rate ({self.unit})")
        ax.set_xlim(min(energy.value), max(energy.value))
        ax.legend(loc="best")

        return ax

    def plot_spectrum(self, ax=None, **kwargs):
        """Plot angle integrated background rate versus energy.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        kwargs : dict
            Forwarded tp plt.plot()

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax
        offset = self.axes["offset"].edges
        energy = self.axes["energy"].center

        bkg = []
        for ee in energy:
            data = self.evaluate(offset=offset, energy=ee)
            val = np.nansum(trapz_loglog(data, offset, axis=0))
            bkg.append(val.value)

        ax.plot(energy, bkg, label="integrated spectrum", **kwargs)

        unit = self.unit * offset.unit ** 2

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"Energy [{energy.unit}]")
        ax.set_ylabel(f"Background rate ({unit})")
        ax.set_xlim(min(energy.value), max(energy.value))
        ax.legend(loc="best")
        return ax

    def peek(self, figsize=(10, 8)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        self.plot(ax=axes[1][1])
        self.plot_offset_dependence(ax=axes[0][0])
        self.plot_energy_dependence(ax=axes[1][0])
        self.plot_spectrum(ax=axes[0][1])
        plt.tight_layout()

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from gammapy.maps import MapAxes, MapAxis
from gammapy.utils.integrate import trapz_loglog
from gammapy.utils.nddata import NDDataArray
from gammapy.utils.scripts import make_path

__all__ = ["Background3D", "Background2D"]

log = logging.getLogger(__name__)


class Background3D:
    """Background 3D.

    Data format specification: :ref:`gadf:bkg_3d`

    Parameters
    ----------
    energy_axis : `MapAxis`
        Energy axis
    fov_lon_axis: `MapAxis`
        FOV coordinate X-axis
    fov_lat_axis : `MapAxis`
        FOV coordinate Y-axis.
    data : `~astropy.units.Quantity`
        Background rate (usually: ``s^-1 MeV^-1 sr^-1``)

    Examples
    --------
    Here's an example you can use to learn about this class:

    >>> from gammapy.irf import Background3D
    >>> filename = '$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits'
    >>> bkg_3d = Background3D.read(filename, hdu='BACKGROUND')
    >>> print(bkg_3d)
    Background3D
    NDDataArray summary info
    energy         : size =    21, min =  0.016 TeV, max = 158.489 TeV
    fov_lon           : size =    36, min = -5.833 deg, max =  5.833 deg
    fov_lat           : size =    36, min = -5.833 deg, max =  5.833 deg
    Data           : size = 27216, min =  0.000 1 / (MeV s sr), max =  0.421 1 / (MeV s sr)
    """

    tag = "bkg_3d"
    default_interp_kwargs = dict(
        bounds_error=False, fill_value=None, values_scale="log"
    )
    """Default Interpolation kwargs for `~gammapy.utils.nddata.NDDataArray`. Extrapolate."""

    def __init__(
        self,
        energy_axis,
        fov_lon_axis,
        fov_lat_axis,
        data,
        meta=None,
        interp_kwargs=None,
    ):
        if interp_kwargs is None:
            interp_kwargs = self.default_interp_kwargs

        self.data = NDDataArray(
            axes=[energy_axis, fov_lon_axis, fov_lat_axis],
            data=data,
            interp_kwargs=interp_kwargs,
        )
        self.meta = meta or {}

    def __str__(self):
        ss = self.__class__.__name__
        ss += f"\n{self.data}"
        return ss

    @classmethod
    def from_table(cls, table):
        """Read from `~astropy.table.Table`."""
        # Spec says key should be "BKG", but there are files around
        # (e.g. CTA 1DC) that use "BGD". For now we support both
        if "BKG" in table.colnames:
            bkg_name = "BKG"
        elif "BGD" in table.colnames:
            bkg_name = "BGD"
        else:
            raise ValueError('Invalid column names. Need "BKG" or "BGD".')

        data_unit = table[bkg_name].unit
        if data_unit is not None:
            data_unit = u.Unit(table[bkg_name].unit, parse_strict="silent")
        if isinstance(data_unit, u.UnrecognizedUnit) or (data_unit is None):
            data_unit = u.Unit("s-1 MeV-1 sr-1")
            log.warning(
                "Invalid unit found in background table! Assuming (s-1 MeV-1 sr-1)"
            )

        energy_axis = MapAxis.from_table(
            table, column_prefix="ENERG", format="gadf-dl3"
        )
        fov_lon_axis = MapAxis.from_table(
            table, column_prefix="DETX", format="gadf-dl3"
        )
        fov_lat_axis = MapAxis.from_table(
            table, column_prefix="DETY", format="gadf-dl3"
        )

        # TODO: The present HESS and CTA backgroundfits files
        #  have a reverse order (lon, lat, E) than recommened in GADF(E, lat, lon)
        #  For now, we suport both.

        data = table[bkg_name].data[0].T * data_unit
        shape = (energy_axis.nbin, fov_lon_axis.nbin, fov_lat_axis.nbin)

        if shape == shape[::-1]:
            log.error("Ambiguous axes order in Background fits files!")

        if np.shape(data) != shape:
            log.debug("Transposing background table on read")
            data = data.transpose()

        return cls(
            energy_axis=energy_axis,
            fov_lon_axis=fov_lon_axis,
            fov_lat_axis=fov_lat_axis,
            data=data,
            meta=table.meta,
        )

    @classmethod
    def from_hdulist(cls, hdulist, hdu="BACKGROUND"):
        """Create from `~astropy.io.fits.HDUList`."""
        return cls.from_table(Table.read(hdulist[hdu]))

    @classmethod
    def read(cls, filename, hdu="BACKGROUND"):
        """Read from file."""
        with fits.open(str(make_path(filename)), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, hdu=hdu)

    def to_table(self):
        """Convert to `~astropy.table.Table`."""
        # TODO: fix axis order
        axes = MapAxes(self.data.axes[::-1])
        table = axes.to_table(format="gadf-dl3")
        table.meta = self.meta.copy()
        table["BKG"] = self.data.data.T[np.newaxis]
        return table

    def to_table_hdu(self, name="BACKGROUND"):
        """Convert to `~astropy.io.fits.BinTableHDU`."""
        return fits.BinTableHDU(self.to_table(), name=name)

    def evaluate(self, fov_lon, fov_lat, energy_reco, method="linear", **kwargs):
        """Evaluate at given FOV position and energy.

        Parameters
        ----------
        fov_lon, fov_lat : `~astropy.coordinates.Angle`
            FOV coordinates expecting in AltAz frame.
        energy_reco : `~astropy.units.Quantity`
            energy on which you want to interpolate. Same dimension than fov_lat and fov_lat
        method : str {'linear', 'nearest'}, optional
            Interpolation method
        kwargs : dict
            option for interpolation for `~scipy.interpolate.RegularGridInterpolator`

        Returns
        -------
        array : `~astropy.units.Quantity`
            Interpolated values, axis order is the same as for the NDData array
        """
        values = self.data.evaluate(
            fov_lon=fov_lon,
            fov_lat=fov_lat,
            energy=energy_reco,
            method=method,
            **kwargs,
        )
        return values

    def evaluate_integrate(
        self, fov_lon, fov_lat, energy_reco, method="linear", **kwargs
    ):
        """Integrate in a given energy band.

        Parameters
        ----------
        fov_lon, fov_lat : `~astropy.coordinates.Angle`
            FOV coordinates expecting in AltAz frame.
        energy_reco: `~astropy.units.Quantity`
            Reconstructed energy edges.
        method : {'linear', 'nearest'}, optional
            Interpolation method

        Returns
        -------
        array : `~astropy.units.Quantity`
            Returns 2D array with axes offset
        """
        data = self.evaluate(fov_lon, fov_lat, energy_reco, method=method)
        return trapz_loglog(data, energy_reco, axis=0)

    def to_2d(self):
        """Convert to `Background2D`.

        This takes the values at Y = 0 and X >= 0.
        """
        # TODO: this is incorrect as it misses the Jacobian?
        idx_lon = self.data.axes["fov_lon"].coord_to_idx(0 * u.deg)[0]
        idx_lat = self.data.axes["fov_lat"].coord_to_idx(0 * u.deg)[0]
        data = self.data.data[:, idx_lon:, idx_lat].copy()

        offset = self.data.axes["fov_lon"].edges[idx_lon:]

        offset_axis = MapAxis.from_edges(offset, name="offset")
        return Background2D(
            energy_axis=self.data.axes["energy"], offset_axis=offset_axis, data=data,
        )

    def peek(self, figsize=(10, 8)):
        return self.to_2d().peek(figsize)


class Background2D:
    """Background 2D.

    Data format specification: :ref:`gadf:bkg_2d`

    Parameters
    ----------
    energy_axis : `MapAxis`
        Energy axis
    offset_axis : `MapAxis`
        FOV coordinate offset-axis
    data : `~astropy.units.Quantity`
        Background rate (usually: ``s^-1 MeV^-1 sr^-1``)
    """

    tag = "bkg_2d"
    default_interp_kwargs = dict(bounds_error=False, fill_value=None)
    """Default Interpolation kwargs for `~gammapy.utils.nddata.NDDataArray`. Extrapolate."""

    def __init__(
        self, energy_axis, offset_axis, data, meta=None, interp_kwargs=None,
    ):
        if interp_kwargs is None:
            interp_kwargs = self.default_interp_kwargs

        assert offset_axis.name == "offset"

        self.data = NDDataArray(
            axes=[energy_axis, offset_axis], data=data, interp_kwargs=interp_kwargs
        )
        self.meta = meta or {}

    def __str__(self):
        ss = self.__class__.__name__
        ss += f"\n{self.data}"
        return ss

    @classmethod
    def from_table(cls, table):
        """Read from `~astropy.table.Table`."""
        # Spec says key should be "BKG", but there are files around
        # (e.g. CTA 1DC) that use "BGD". For now we support both

        if "BKG" in table.colnames:
            bkg_name = "BKG"
        elif "BGD" in table.colnames:
            bkg_name = "BGD"
        else:
            raise ValueError('Invalid column names. Need "BKG" or "BGD".')

        data_unit = table[bkg_name].unit
        if data_unit is not None:
            data_unit = u.Unit(data_unit, parse_strict="silent")
        if isinstance(data_unit, u.UnrecognizedUnit) or (data_unit is None):
            data_unit = u.Unit("s-1 MeV-1 sr-1")
            log.warning(
                "Invalid unit found in background table! Assuming (s-1 MeV-1 sr-1)"
            )

        energy_axis = MapAxis.from_table(
            table, column_prefix="ENERG", format="gadf-dl3"
        )
        offset_axis = MapAxis.from_table(
            table, column_prefix="THETA", format="gadf-dl3"
        )

        # TODO: The present HESS and CTA backgroundfits files
        # have a reverse order (theta, E) than recommened in GADF(E, theta)
        # For now, we suport both.

        data = table[bkg_name].data[0].T * data_unit
        shape = (energy_axis.nbin, offset_axis.nbin)

        if shape == shape[::-1]:
            log.error("Ambiguous axes order in Background fits files!")

        if np.shape(data) != shape:
            log.debug("Transposing background table on read")
            data = data.transpose()

        return cls(
            energy_axis=energy_axis,
            offset_axis=offset_axis,
            data=data,
            meta=table.meta,
        )

    @classmethod
    def from_hdulist(cls, hdulist, hdu="BACKGROUND"):
        """Create from `~astropy.io.fits.HDUList`."""
        return cls.from_table(Table.read(hdulist[hdu]))

    @classmethod
    def read(cls, filename, hdu="BACKGROUND"):
        """Read from file."""
        with fits.open(str(make_path(filename)), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, hdu=hdu)

    def to_table(self):
        """Convert to `~astropy.table.Table`."""
        table = self.data.axes.to_table(format="gadf-dl3")
        table.meta = self.meta.copy()
        table["BKG"] = self.data.data.T[np.newaxis]
        return table

    def to_table_hdu(self, name="BACKGROUND"):
        """Convert to `~astropy.io.fits.BinTableHDU`."""
        return fits.BinTableHDU(self.to_table(), name=name)

    def evaluate(self, fov_lon, fov_lat, energy_reco, method="linear", **kwargs):
        """Evaluate at a given FOV position and energy.

        The fov_lon, fov_lat, energy_reco has to have the same shape
        since this is a set of points on which you want to evaluate.

        To have the same API than background 3D for the
        background evaluation, the offset is ``fov_altaz_lon``.

        Parameters
        ----------
        fov_lon, fov_lat : `~astropy.coordinates.Angle`
            FOV coordinates expecting in AltAz frame, same shape than energy_reco
        energy_reco : `~astropy.units.Quantity`
            Reconstructed energy, same dimension than fov_lat and fov_lat
        method : str {'linear', 'nearest'}, optional
            Interpolation method
        kwargs : dict
            option for interpolation for `~scipy.interpolate.RegularGridInterpolator`

        Returns
        -------
        array : `~astropy.units.Quantity`
            Interpolated values, axis order is the same as for the NDData array
        """
        offset = np.sqrt(fov_lon ** 2 + fov_lat ** 2)
        return self.data.evaluate(
            offset=offset, energy=energy_reco, method=method, **kwargs
        )

    def evaluate_integrate(self, fov_lon, fov_lat, energy_reco, method="linear"):
        """Evaluate at given FOV position and energy, by integrating over the energy range.

        Parameters
        ----------
        fov_lon, fov_lat : `~astropy.coordinates.Angle`
            FOV coordinates expecting in AltAz frame.
        energy_reco: `~astropy.units.Quantity`
            Reconstructed energy edges.
        method : {'linear', 'nearest'}, optional
            Interpolation method

        Returns
        -------
        array : `~astropy.units.Quantity`
            Returns 2D array with axes offset
        """
        data = self.evaluate(fov_lon, fov_lat, energy_reco, method=method)
        return trapz_loglog(data, energy_reco, axis=0)

    def to_3d(self):
        """Convert to `Background3D`.

        Fill in a radially symmetric way.
        """
        raise NotImplementedError

    def plot(self, ax=None, add_cbar=True, **kwargs):
        """Plot energy offset dependence of the background model.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        ax = plt.gca() if ax is None else ax

        x = self.data.axes["energy"].edges.to_value("TeV")
        y = self.data.axes["offset"].edges.to_value("deg")
        z = self.data.data.T.value

        kwargs.setdefault("cmap", "GnBu")
        kwargs.setdefault("edgecolors", "face")

        caxes = ax.pcolormesh(x, y, z, norm=LogNorm(), **kwargs)
        ax.set_xscale("log")
        ax.set_ylabel(f"Offset (deg)")
        ax.set_xlabel(f"Energy (TeV)")

        xmin, xmax = x.min(), x.max()
        ax.set_xlim(xmin, xmax)

        if add_cbar:
            label = f"Background rate ({self.data.data.unit})"
            ax.figure.colorbar(caxes, ax=ax, label=label)

    def plot_offset_dependence(self, ax=None, offset=None, energy=None, **kwargs):
        """Plot background rate versus offset for a given energy.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        offset : `~astropy.coordinates.Angle`
            Offset axis
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
            energy_axis = self.data.axes["energy"]
            e_min, e_max = np.log10(energy_axis.center.value[[0, -1]])
            energy = np.logspace(e_min, e_max, 4) * energy_axis.unit

        if offset is None:
            offset = self.data.axes["offset"].center

        for ee in energy:
            bkg = self.data.evaluate(offset=offset, energy=ee)
            if np.isnan(bkg).all():
                continue
            label = f"energy = {ee:.1f}"
            ax.plot(offset, bkg.value, label=label, **kwargs)

        ax.set_xlabel(f"Offset ({self.data.axes['offset'].unit})")
        ax.set_ylabel(f"Background rate ({self.data.data.unit})")
        ax.set_yscale("log")
        ax.legend(loc="upper right")
        return ax

    def plot_energy_dependence(self, ax=None, offset=None, energy=None, **kwargs):
        """Plot background rate versus energy for a given offset.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        offset : `~astropy.coordinates.Angle`
            Offset
        energy : `~astropy.units.Quantity`
            Energy axis
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
            offset_axis = self.data.axes["offset"]
            off_min, off_max = offset_axis.center.value[[0, -1]]
            offset = np.linspace(off_min, off_max, 4) * offset_axis.unit

        if energy is None:
            energy = self.data.axes["energy"].center

        for off in offset:
            bkg = self.data.evaluate(offset=off, energy=energy)
            kwargs.setdefault("label", f"offset = {off:.1f}")
            ax.plot(energy, bkg.value, **kwargs)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"Energy [{energy.unit}]")
        ax.set_ylabel(f"Background rate ({self.data.data.unit})")
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
        offset = self.data.axes["offset"].edges
        energy = self.data.axes["energy"].center

        bkg = []
        for ee in energy:
            data = self.data.evaluate(offset=offset, energy=ee)
            val = np.nansum(trapz_loglog(data, offset, axis=0))
            bkg.append(val.value)

        ax.plot(energy, bkg, label="integrated spectrum", **kwargs)

        unit = self.data.data.unit * offset.unit * offset.unit

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

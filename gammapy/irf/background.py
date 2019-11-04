# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from gammapy.maps import MapAxis
from gammapy.maps.utils import edges_from_lo_hi
from gammapy.utils.integrate import _trapz_loglog
from gammapy.utils.nddata import NDDataArray
from gammapy.utils.scripts import make_path

__all__ = ["Background3D", "Background2D"]


class Background3D:
    """Background 3D.

    Data format specification: :ref:`gadf:bkg_3d`

    Parameters
    ----------
    energy_lo, energy_hi : `~astropy.units.Quantity`
        Energy binning
    fov_lon_lo, fov_lon_hi : `~astropy.units.Quantity`
        FOV coordinate X-axis binning.
    fov_lat_lo, fov_lat_hi : `~astropy.units.Quantity`
        FOV coordinate Y-axis binning.
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

    default_interp_kwargs = dict(
        bounds_error=False, fill_value=None, values_scale="log"
    )
    """Default Interpolation kwargs for `~gammapy.utils.nddata.NDDataArray`. Extrapolate."""

    def __init__(
        self,
        energy_lo,
        energy_hi,
        fov_lon_lo,
        fov_lon_hi,
        fov_lat_lo,
        fov_lat_hi,
        data,
        meta=None,
        interp_kwargs=None,
    ):
        if interp_kwargs is None:
            interp_kwargs = self.default_interp_kwargs

        e_edges = edges_from_lo_hi(energy_lo, energy_hi)
        energy_axis = MapAxis.from_edges(e_edges, interp="log", name="energy")

        fov_lon_edges = edges_from_lo_hi(fov_lon_lo, fov_lon_hi)
        fov_lon_axis = MapAxis.from_edges(fov_lon_edges, interp="lin", name="fov_lon")

        fov_lat_edges = edges_from_lo_hi(fov_lat_lo, fov_lat_hi)
        fov_lat_axis = MapAxis.from_edges(fov_lat_edges, interp="lin", name="fov_lat")

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

        # Currently some files (e.g. CTA 1DC) contain unit in the FITS file
        # '1/s/MeV/sr', which is invalid ( try: astropy.units.Unit('1/s/MeV/sr')
        # This should be corrected.
        # For now, we hard-code the unit here:
        data_unit = u.Unit("s-1 MeV-1 sr-1")

        return cls(
            energy_lo=table["ENERG_LO"].quantity[0],
            energy_hi=table["ENERG_HI"].quantity[0],
            fov_lon_lo=table["DETX_LO"].quantity[0],
            fov_lon_hi=table["DETX_HI"].quantity[0],
            fov_lat_lo=table["DETY_LO"].quantity[0],
            fov_lat_hi=table["DETY_HI"].quantity[0],
            data=table[bkg_name].data[0] * data_unit,
            meta=table.meta,
        )

    @classmethod
    def from_hdulist(cls, hdulist, hdu="BACKGROUND"):
        """Create from `~astropy.io.fits.HDUList`."""
        return cls.from_table(Table.read(hdulist[hdu]))

    @classmethod
    def read(cls, filename, hdu="BACKGROUND"):
        """Read from file."""
        with fits.open(make_path(filename), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, hdu=hdu)

    def to_table(self):
        """Convert to `~astropy.table.Table`."""
        meta = self.meta.copy()

        detx = self.data.axis("fov_lon").edges
        dety = self.data.axis("fov_lat").edges
        energy = self.data.axis("energy").edges

        table = Table(meta=meta)
        table["DETX_LO"] = detx[:-1][np.newaxis]
        table["DETX_HI"] = detx[1:][np.newaxis]
        table["DETY_LO"] = dety[:-1][np.newaxis]
        table["DETY_HI"] = dety[1:][np.newaxis]
        table["ENERG_LO"] = energy[:-1][np.newaxis]
        table["ENERG_HI"] = energy[1:][np.newaxis]
        table["BKG"] = self.data.data[np.newaxis]
        return table

    def to_fits(self, name="BACKGROUND"):
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
        return _trapz_loglog(data, energy_reco, axis=0, intervals=True)

    def to_2d(self):
        """Convert to `Background2D`.

        This takes the values at Y = 0 and X >= 0.
        """
        idx_lon = self.data.axis("fov_lon").coord_to_idx(0 * u.deg)[0]
        idx_lat = self.data.axis("fov_lat").coord_to_idx(0 * u.deg)[0]
        data = self.data.data[:, idx_lon:, idx_lat].copy()

        energy = self.data.axis("energy").edges
        offset = self.data.axis("fov_lon").edges[idx_lon:]

        return Background2D(
            energy_lo=energy[:-1],
            energy_hi=energy[1:],
            offset_lo=offset[:-1],
            offset_hi=offset[1:],
            data=data,
        )


class Background2D:
    """Background 2D.

    Data format specification: :ref:`gadf:bkg_2d`

    Parameters
    ----------
    energy_lo, energy_hi : `~astropy.units.Quantity`
        Energy binning
    offset_lo, offset_hi : `~astropy.units.Quantity`
        FOV coordinate offset-axis binning
    data : `~astropy.units.Quantity`
        Background rate (usually: ``s^-1 MeV^-1 sr^-1``)
    """

    default_interp_kwargs = dict(bounds_error=False, fill_value=None)
    """Default Interpolation kwargs for `~gammapy.utils.nddata.NDDataArray`. Extrapolate."""

    def __init__(
        self,
        energy_lo,
        energy_hi,
        offset_lo,
        offset_hi,
        data,
        meta=None,
        interp_kwargs=None,
    ):
        if interp_kwargs is None:
            interp_kwargs = self.default_interp_kwargs

        e_edges = edges_from_lo_hi(energy_lo, energy_hi)
        energy_axis = MapAxis.from_edges(e_edges, interp="log", name="energy")

        offset_edges = edges_from_lo_hi(offset_lo, offset_hi)
        offset_axis = MapAxis.from_edges(offset_edges, interp="lin", name="offset")

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

        # Currently some files (e.g. CTA 1DC) contain unit in the FITS file
        # '1/s/MeV/sr', which is invalid ( try: astropy.units.Unit('1/s/MeV/sr')
        # This should be corrected.
        # For now, we hard-code the unit here:
        data_unit = u.Unit("s-1 MeV-1 sr-1")
        return cls(
            energy_lo=table["ENERG_LO"].quantity[0],
            energy_hi=table["ENERG_HI"].quantity[0],
            offset_lo=table["THETA_LO"].quantity[0],
            offset_hi=table["THETA_HI"].quantity[0],
            data=table[bkg_name].data[0] * data_unit,
            meta=table.meta,
        )

    @classmethod
    def from_hdulist(cls, hdulist, hdu="BACKGROUND"):
        """Create from `~astropy.io.fits.HDUList`."""
        return cls.from_table(Table.read(hdulist[hdu]))

    @classmethod
    def read(cls, filename, hdu="BACKGROUND"):
        """Read from file."""
        with fits.open(make_path(filename), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, hdu=hdu)

    def to_table(self):
        """Convert to `~astropy.table.Table`."""
        meta = self.meta.copy()
        table = Table(meta=meta)

        theta = self.data.axis("offset").edges
        energy = self.data.axis("energy").edges

        table["THETA_LO"] = theta[:-1][np.newaxis]
        table["THETA_HI"] = theta[1:][np.newaxis]
        table["ENERG_LO"] = energy[:-1][np.newaxis]
        table["ENERG_HI"] = energy[1:][np.newaxis]
        table["BKG"] = self.data.data[np.newaxis]
        return table

    def to_fits(self, name="BACKGROUND"):
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
        return _trapz_loglog(data, energy_reco, axis=0, intervals=True)

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

        x = self.data.axis("energy").edges.to_value("TeV")
        y = self.data.axis("offset").edges.to_value("deg")
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

    def peek(self):
        from .effective_area import EffectiveAreaTable2D

        return EffectiveAreaTable2D.peek(self)

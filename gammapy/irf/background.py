# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
from astropy.table import Table
from astropy.io import fits
import astropy.units as u
from ..utils.nddata import NDDataArray, BinnedDataAxis
from ..utils.scripts import make_path
from ..utils.energy import EnergyBounds

__all__ = ["Background3D", "Background2D"]


class Background3D(object):
    """Background 3D.

    Data format specification: :ref:`gadf:bkg_3d`

    Parameters
    -----------
    energy_lo, energy_hi : `~astropy.units.Quantity`
        Energy binning
    fov_lon_lo, fov_lon_hi : `~astropy.units.Quantity`
        FOV coordinate X-axis binning, in AltAz frame.
    fov_lat_lo, fov_lat_hi : `~astropy.units.Quantity`
        FOV coordinate Y-axis binning, in AltAz frame.
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

    default_interp_kwargs = dict(bounds_error=False, fill_value=None)
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
        axes = [
            BinnedDataAxis(
                energy_lo, energy_hi, interpolation_mode="log", name="energy"
            ),
            BinnedDataAxis(
                fov_lon_lo, fov_lon_hi, interpolation_mode="linear", name="fov_lon"
            ),
            BinnedDataAxis(
                fov_lat_lo, fov_lat_hi, interpolation_mode="linear", name="fov_lat"
            ),
        ]
        self.data = NDDataArray(axes=axes, data=data, interp_kwargs=interp_kwargs)
        self.meta = OrderedDict(meta) if meta else OrderedDict()

    def __str__(self):
        ss = self.__class__.__name__
        ss += "\n{}".format(self.data)
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
        # '1/s/MeV/sr', which is invalid ( try: astropy.unit.Unit('1/s/MeV/sr')
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
        filename = make_path(filename)
        with fits.open(str(filename), memmap=False) as hdulist:
            bkg = cls.from_hdulist(hdulist, hdu=hdu)

        return bkg

    def to_table(self):
        """Convert to `~astropy.table.Table`."""
        meta = self.meta.copy()
        table = Table(meta=meta)
        table["DETX_LO"] = self.data.axis("fov_lon").lo[np.newaxis]
        table["DETX_HI"] = self.data.axis("fov_lon").hi[np.newaxis]
        table["DETY_LO"] = self.data.axis("fov_lat").lo[np.newaxis]
        table["DETY_HI"] = self.data.axis("fov_lat").hi[np.newaxis]
        table["ENERG_LO"] = self.data.axis("energy").lo[np.newaxis]
        table["ENERG_HI"] = self.data.axis("energy").hi[np.newaxis]
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
        points = dict(fov_lon=fov_lon, fov_lat=fov_lat, energy=energy_reco)
        array = self.data.evaluate_at_coord(points=points, method=method, **kwargs)
        return array

    def integrate_on_energy_range(
        self,
        fov_lon,
        fov_lat,
        energy_range,
        n_integration_bins=1,
        method="linear",
        **kwargs
    ):
        """Integrate over an energy range.

        Parameters
        ----------
        fov_lon, fov_lat : `~astropy.coordinates.Angle`
            FOV coordinates expecting in AltAz frame.
        energy_range: `~astropy.units.Quantity`
            Energy range
        n_integration_bins : int
            Number of bins in the energy range
        method : {'linear', 'nearest'}, optional
            Interpolation method
        kwargs : dict
            Passed to `scipy.interpolate.RegularGridInterpolator`.

        Returns
        -------
        array : `~astropy.units.Quantity`
            Returns 2D array with axes fov_lon, fov_lat
        """
        fov_lon = np.atleast_2d(fov_lon)
        fov_lat = np.atleast_2d(fov_lat)
        energy_edges = EnergyBounds.equal_log_spacing(
            energy_range[0], energy_range[1], n_integration_bins
        )

        # TODO: insert new axes, remove tile and use numpy broadcasting
        energy_reco = np.tile(energy_edges, reps=fov_lon.shape + (1,))
        fov_lon = np.tile(fov_lon, reps=energy_edges.shape + (1, 1))
        fov_lon = np.rollaxis(fov_lon, 0, 3)
        fov_lat = np.tile(fov_lat, reps=energy_edges.shape + (1, 1))
        fov_lat = np.rollaxis(fov_lat, 0, 3)

        bkg_evaluated = self.evaluate(
            fov_lon=fov_lon,
            fov_lat=fov_lat,
            energy_reco=energy_reco,
            method=method,
            **kwargs
        )
        # TODO: use gammapy.spectrum.utils._trapz_loglog for better precision
        return np.trapz(bkg_evaluated, energy_edges).decompose()

    def to_2d(self):
        """Convert to `Background2D`.

        This takes the values at Y = 0 and X >= 0.
        """
        idx_lon = self.data.axis("fov_lon").find_node("0 deg")[0]
        idx_lat = self.data.axis("fov_lat").find_node("0 deg")[0]
        data = self.data.data[:, idx_lon:, idx_lat].copy()

        return Background2D(
            energy_lo=self.data.axis("energy").lo,
            energy_hi=self.data.axis("energy").hi,
            offset_lo=self.data.axis("fov_lon").lo[idx_lon:],
            offset_hi=self.data.axis("fov_lon").hi[idx_lon:],
            data=data,
        )


class Background2D(object):
    """Background 2D.

    Data format specification: :ref:`gadf:bkg_2d`

    Parameters
    -----------
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
        axes = [
            BinnedDataAxis(
                energy_lo, energy_hi, interpolation_mode="log", name="energy"
            ),
            BinnedDataAxis(
                offset_lo, offset_hi, interpolation_mode="linear", name="offset"
            ),
        ]
        self.data = NDDataArray(axes=axes, data=data, interp_kwargs=interp_kwargs)
        self.meta = OrderedDict(meta) if meta else OrderedDict()

    def __str__(self):
        ss = self.__class__.__name__
        ss += "\n{}".format(self.data)
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
        # '1/s/MeV/sr', which is invalid ( try: astropy.unit.Unit('1/s/MeV/sr')
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
        filename = make_path(filename)
        with fits.open(str(filename), memmap=False) as hdulist:
            bkg = cls.from_hdulist(hdulist, hdu=hdu)

        return bkg

    def to_table(self):
        """Convert to `~astropy.table.Table`."""
        meta = self.meta.copy()
        table = Table(meta=meta)

        table["THETA_LO"] = self.data.axis("offset").lo[np.newaxis]
        table["THETA_HI"] = self.data.axis("offset").hi[np.newaxis]
        table["ENERG_LO"] = self.data.axis("energy").lo[np.newaxis]
        table["ENERG_HI"] = self.data.axis("energy").hi[np.newaxis]
        table["BKG"] = self.data.data[np.newaxis]
        return table

    def to_fits(self, name="BACKGROUND"):
        """Convert to `~astropy.io.fits.BinTableHDU`."""
        return fits.BinTableHDU(self.to_table(), name=name)

    def evaluate(self, fov_lon, fov_lat, energy_reco, method="linear", **kwargs):
        """Evaluate at a given FOV position and energy. The fov_lon, fov_lat, energy_reco has to have the same shape
        since this is a set of points on which you want to evaluate

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
        points = dict(offset=offset, energy=energy_reco)
        return self.data.evaluate_at_coord(points=points, method=method, **kwargs)

    def integrate_on_energy_range(
        self,
        fov_lon,
        fov_lat,
        energy_range,
        n_integration_bins=1,
        method="linear",
        **kwargs
    ):
        """Integrate over an energy range.

        Parameters
        ----------
        fov_lon, fov_lat : `~astropy.coordinates.Angle`
            FOV coordinates expecting in AltAz frame.
        energy_range: `~astropy.units.Quantity`
            Energy range
        n_integration_bins : int
            Number of bins in the energy range
        method : {'linear', 'nearest'}, optional
            Interpolation method
        kwargs : dict
            Passed to `scipy.interpolate.RegularGridInterpolator`.

        Returns
        -------
        array : `~astropy.units.Quantity`
            Returns 2D array with axes offset
        """
        fov_lon = np.atleast_2d(fov_lon)
        fov_lat = np.atleast_2d(fov_lat)
        energy_edges = EnergyBounds.equal_log_spacing(
            energy_range[0], energy_range[1], n_integration_bins
        )
        # TODO: insert new axes, remove tile and use numpy broadcasting
        energy_reco = np.tile(energy_edges, reps=fov_lon.shape + (1,))
        fov_lon = np.tile(fov_lon, reps=energy_edges.shape + (1, 1))
        fov_lon = np.rollaxis(fov_lon, 0, 3)
        fov_lat = np.tile(fov_lat, reps=energy_edges.shape + (1, 1))
        fov_lat = np.rollaxis(fov_lat, 0, 3)

        bkg_evaluated = self.evaluate(
            fov_lon=fov_lon,
            fov_lat=fov_lat,
            energy_reco=energy_reco,
            method=method,
            **kwargs
        )

        # TODO: use gammapy.spectrum.utils._trapz_loglog for better precision
        return np.trapz(bkg_evaluated, energy_edges).decompose()

    def to_3d(self):
        """Convert to `Background3D`.

        Fill in a radially symmetric way.
        """
        raise NotImplementedError

    def plot(self, **kwargs):
        from .effective_area import EffectiveAreaTable2D

        return EffectiveAreaTable2D.plot(self, **kwargs)

    def peek(self):
        from .effective_area import EffectiveAreaTable2D

        return EffectiveAreaTable2D.peek(self)

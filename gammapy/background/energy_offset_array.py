# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity
from astropy.table import Table
from ..utils.energy import EnergyBounds, Energy
from ..utils.scripts import make_path
from .fov_cube import _make_bin_edges_array, FOVCube

__all__ = [
    'EnergyOffsetArray',
]


class EnergyOffsetArray(object):
    """Energy offset dependent array.

    TODO: take quantity `data` in `__init__` instead of `data` and `data_units` separately.

    Parameters
    ----------
    energy : `~gammapy.utils.energy.EnergyBounds`
        Energy bounds vector (1D)
    offset : `~astropy.coordinates.Angle`
        Offset vector (1D)
    data : `~numpy.ndarray`, optional
        Data array (2D)
    data_err : `~numpy.ndarray`, optional
        Data array (2D) containing the errors on the data
    """

    def __init__(self, energy, offset, data=None, data_units="", data_err=None):
        self.energy = EnergyBounds(energy)
        self.offset = Angle(offset)
        if data is None:
            self.data = Quantity(np.zeros((len(energy) - 1, len(offset) - 1)), data_units)
        else:
            self.data = Quantity(data, data_units)
        if data_err is None:
            self.data_err = None
        else:
            self.data_err = Quantity(data_err, data_units)

    def fill_events(self, event_lists):
        """Fill events histogram.

        This add the counts to the existing value array.

        Parameters
        ----------
        event_lists : list of `~gammapy.data.EventList`
           Python list of event list objects.
        """
        for event_list in event_lists:
            counts = self._fill_one_event_list(event_list)
            self.data += Quantity(counts, unit=self.data.unit)
        self.data_err = np.sqrt(self.data)

    def _fill_one_event_list(self, events):
        """Histogram the counts of an EventList object in 2D (energy,offset).

        Parameters
        ----------
        events :`~gammapy.data.EventList`
           Event list
        """
        offset = events.offset.to(self.offset.unit).value
        ev_energy = events.energy.to(self.energy.unit).value

        sample = np.vstack([ev_energy, offset]).T
        bins = [self.energy.value, self.offset.value]
        hist, edges = np.histogramdd(sample, bins)

        return hist

    def plot(self, ax=None, **kwargs):
        """Plot Energy_offset Array image (x=offset, y=energy).
        """
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        offset, energy = self.offset, self.energy.log_centers
        x, y, z = offset.value, energy.value, self.data.value
        caxes = ax.pcolormesh(x, y, z, **kwargs)
        unit = self.data.unit
        cbar = ax.figure.colorbar(caxes, ax=ax, label='Value ({})'.format(unit))
        ax.semilogy()
        ax.set_xlabel('Offset ({})'.format(offset.unit))
        ax.set_ylabel('Energy ({})'.format(energy.unit))

        return ax, cbar

    @classmethod
    def read(cls, filename, hdu='bkg_2d', data_name="data"):
        """Read from  FITS file.

        Parameters
        ----------
        filename : str
            File name
        hdu : str
            HDU name
        data_name : str
            Name of the data column in the table
        """
        filename = make_path(filename)
        table = Table.read(str(filename), hdu=hdu, format='fits')
        return cls.from_table(table, data_name)

    @classmethod
    def from_table(cls, table, data_name="data"):
        """Create from `~astropy.table.Table`.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table
        data_name : str
            Name of the data column in the table
        """
        offset_edges = _make_bin_edges_array(table['THETA_LO'].squeeze(), table['THETA_HI'].squeeze())
        offset_edges = Angle(offset_edges, table['THETA_LO'].unit)
        energy_edges = _make_bin_edges_array(table['ENERG_LO'].squeeze(), table['ENERG_HI'].squeeze())
        energy_edges = EnergyBounds(energy_edges, table['ENERG_LO'].unit)
        data = Quantity(table[data_name].squeeze(), table[data_name].unit)
        if data_name + "_err" in table.colnames:
            data_err = Quantity(table[data_name + "_err"].squeeze(), table[data_name + "_err"].unit)
        else:
            data_err = None

        return cls(energy_edges, offset_edges, data, data_units=str(data.unit), data_err=data_err)

    def write(self, filename, data_name="data", **kwargs):
        """Write to FITS file.

        Parameters
        ----------
        filename : str
            File name
        data_name : str
            Name of the data column in the table
        """
        self.to_table(data_name).write(filename, **kwargs)

    def to_table(self, data_name="data"):
        """Convert to `~astropy.table.Table`.

        Parameters
        ----------
        data_name : str
            Name of the data column in the table

        Returns
        -------
        table : `~astropy.table.Table`
            Table containing the EnergyOffsetArray.
        """
        table = Table()
        table['THETA_LO'] = Quantity([self.offset[:-1]], unit=self.offset.unit)
        table['THETA_HI'] = Quantity([self.offset[1:]], unit=self.offset.unit)
        table['ENERG_LO'] = Quantity([self.energy[:-1]], unit=self.energy.unit)
        table['ENERG_HI'] = Quantity([self.energy[1:]], unit=self.energy.unit)
        table[data_name] = Quantity([self.data], unit=self.data.unit)
        if self.data_err is not None:
            table[data_name + "_err"] = Quantity([self.data_err], unit=self.data_err.unit)

        return table

    def evaluate(self, energy=None, offset=None,
                 interp_kwargs=None):
        """Interpolate at a given offset and energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            energy value
        offset : `~astropy.coordinates.Angle`
            offset value
        interp_kwargs : dict
            option for interpolation for `~scipy.interpolate.RegularGridInterpolator`

        Returns
        -------
        values : `~astropy.units.Quantity`
            Interpolated value
        """
        from scipy.interpolate import RegularGridInterpolator
        if not interp_kwargs:
            interp_kwargs = dict(bounds_error=False, fill_value=None)

        if energy is None:
            energy = self.energy.log_centers
        if offset is None:
            offset = self.offset_bin_center

        energy = Energy(energy).to('TeV')
        offset = Angle(offset).to('deg')

        energy_bin = self.energy.to('TeV').log_centers
        offset_bin = self.offset_bin_center
        points = (energy_bin, offset_bin)
        interpolator = RegularGridInterpolator(points, self.data.value, **interp_kwargs)
        ee, off = np.meshgrid(energy.value, offset.value, indexing='ij')
        shape = ee.shape
        pix_coords = np.column_stack([ee.flat, off.flat])

        data_interp = interpolator(pix_coords)
        return Quantity(data_interp.reshape(shape), self.data.unit)

    @property
    def offset_bin_center(self):
        """Offset bin center location (1D `~astropy.coordinates.Angle` in deg)."""
        off = (self.offset[:-1] + self.offset[1:]) / 2.
        return off.to("deg")

    @property
    def solid_angle(self):
        """Solid angle for each offset bin (1D `~astropy.units.Quantity` in sr)."""
        s = np.pi * (self.offset[1:] ** 2 - self.offset[:-1] ** 2)
        return s.to('sr')

    @property
    def bin_volume(self):
        """Per-pixel bin volume (solid angle * energy). (2D `~astropy.units.Quantity` in Tev sr)."""
        delta_energy = self.energy.bands
        solid_angle = self.solid_angle
        # define grid of deltas (i.e. bin widths for each 3D bin)
        delta_energy, solid_angle = np.meshgrid(delta_energy, solid_angle, indexing='ij')
        bin_volume = delta_energy * solid_angle

        return bin_volume.to('TeV sr')

    def evaluate_at_energy(self, energy, interp_kwargs=None):
        """Evaluate at one given energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        interp_kwargs : dict
            Option for interpolation for `~scipy.interpolate.RegularGridInterpolator`

        Returns
        -------
        table : `~astropy.table.Table`
            Table with two columns: offset, value
        """
        table = Table()
        table["offset"] = self.offset_bin_center
        table["value"] = self.evaluate(energy, None, interp_kwargs)[0, :]
        return table

    def evaluate_at_offset(self, offset, interp_kwargs=None):
        """Evaluate at one given offset.

        Parameters
        ----------
        offset : `~astropy.coordinates.Angle`
            Offset angle
        interp_kwargs : dict
            option for interpolation for `~scipy.interpolate.RegularGridInterpolator`

        Returns
        -------
        table : `~astropy.table.Table`
            Table with two columns: energy, value
        """
        table = Table()
        table["energy"] = self.energy.log_centers
        table["value"] = self.evaluate(None, offset, interp_kwargs)[:, 0]
        return table

    def acceptance_curve_in_energy_band(self, energy_band, energy_bins=10, interp_kwargs=None):
        """Compute acceptance curve in energy band.

        Evaluate the `EnergyOffsetArray` at different energies in the energy_band.
        Then integrate them in order to get the total acceptance curve

        Parameters
        ----------
        energy_band : `~astropy.units.Quantity`
            Tuple ``(energy_min, energy_max)``
        energy_bins : int or `~astropy.units.Quantity`
            Energy bin definition.
        interp_kwargs : dict
            option for interpolation for `~scipy.interpolate.RegularGridInterpolator`

        Returns
        -------
        table : `~astropy.table.Table`
            two column: offset and integral values (units = self.data.unit * self.energy.units)

        """
        [emin, emax] = energy_band
        energy_edges = EnergyBounds.equal_log_spacing(emin, emax, energy_bins)
        energy_bins = energy_edges.log_centers
        acceptance = self.evaluate(energy_bins, None, interp_kwargs)
        # Sum over the energy (axis=1 since we used .T to broadcast acceptance and energy_edges.bands
        acceptance_tot = np.sum(acceptance.T * energy_edges.bands, axis=1)
        table = Table()
        table["offset"] = self.offset_bin_center
        table["Acceptance"] = acceptance_tot.decompose()
        return table

    def to_cube(self, coordx_edges=None, coordy_edges=None, energy_edges=None, interp_kwargs=None):
        """Transform into a `FOVCube`.

        Parameters
        ----------
        coordx_edges : `~astropy.coordinates.Angle`, optional
            Spatial bin edges vector (low and high). X coordinate.
        coordy_edges : `~astropy.coordinates.Angle`, optional
            Spatial bin edges vector (low and high). Y coordinate.
        energy_edges : `~gammapy.utils.energy.EnergyBounds`, optional
            Energy bin edges vector (low and high).
        interp_kwargs : dict
            option for interpolation for `~scipy.interpolate.RegularGridInterpolator`

        Returns
        -------
        cube : `~gammapy.background.FOVCube`
            FOVCube
        """
        if coordx_edges is None:
            offmax = self.offset.max() / 2.
            offmin = self.offset.min()
            nbin = 2 * len(self.offset)
            coordx_edges = np.linspace(offmax, offmin, nbin)
        if coordy_edges is None:
            offmax = self.offset.max() / 2.
            offmin = self.offset.min()
            nbin = 2 * len(self.offset)
            coordy_edges = np.linspace(offmax, offmin, nbin)
        if energy_edges is None:
            energy_edges = self.energy

        coordx_center = (coordx_edges[:-1] + coordx_edges[1:]) / 2.
        coordy_center = (coordy_edges[:-1] + coordy_edges[1:]) / 2.

        xx, yy = np.meshgrid(coordx_center, coordy_center)
        dist = np.sqrt(xx ** 2 + yy ** 2)
        shape = np.shape(dist)
        data = self.evaluate(energy_edges.log_centers, dist.flat, interp_kwargs)

        data_reshape = np.zeros((len(energy_edges) - 1,
                                 len(coordy_edges) - 1,
                                 len(coordx_edges) - 1))
        for i in range(len(energy_edges.log_centers)):
            data_reshape[i, :, :] = np.reshape(data[i, :], shape)
        return FOVCube(coordx_edges, coordy_edges, energy_edges, data_reshape)

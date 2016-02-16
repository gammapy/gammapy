# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity
from astropy.table import Table
from astropy.io import fits
from ..utils.fits import table_to_fits_table
from ..utils.energy import EnergyBounds, Energy
from .cube import _make_bin_edges_array

__all__ = [
    'EnergyOffsetArray',
]


class EnergyOffsetArray(object):
    """
    Energy offset dependent array.

    Parameters
    ----------
    energy : `~gammapy.utils.energy.EnergyBounds`
         energy bin vector
    offset : `~astropy.coordinates.Angle`
        offset bin vector
    data : `~numpy.ndarray`
        data array (2D)
    """

    def __init__(self, energy, offset, data=None, data_units=""):
        self.energy = EnergyBounds(energy)
        self.offset = Angle(offset)
        if data is None:
            self.data = Quantity(np.zeros((len(energy) - 1, len(offset) - 1)), data_units)
        else:
            self.data = Quantity(data, data_units)

    def fill_events(self, event_lists):
        """Fill events histogram.

        This add the counts to the existing value array.

        Parameters
        -------------
        event_lists : list of `~gammapy.data.EventList`
           Python list of event list objects.
        """
        for event_list in event_lists:
            counts = self._fill_one_event_list(event_list)
            self.data += Quantity(counts, unit=self.data.unit)

    def _fill_one_event_list(self, events):
        """
        histogram the counts of an EventList object in 2D (energy,offset)

        Parameters
        -------------
        events :`~gammapy.data.EventList`
           Event list objects.
        """
        offset = events.offset
        ev_energy = events.energy

        sample = np.vstack([ev_energy, offset]).T
        bins = [self.energy.value, self.offset.value]
        hist, edges = np.histogramdd(sample, bins)

        return hist

    def plot_image(self, ax=None, offset=None, energy=None, **kwargs):
        """
        Plot Energy_offset Array image (x=offset, y=energy).
        """
        import matplotlib.pyplot as plt

        kwargs.setdefault('cmap', 'afmhot')
        kwargs.setdefault('origin', 'bottom')
        kwargs.setdefault('interpolation', 'nearest')

        ax = plt.gca() if ax is None else ax

        if offset is None:
            offset = self.offset

        if energy is None:
            energy = self.energy

        extent = [
            offset.value.min(), offset.value.max(),
            energy.value.min(), energy.value.max(),
        ]
        ax.imshow(self.data.value, extent=extent, **kwargs)
        ax.semilogy()
        ax.set_xlabel('Offset ({0})'.format(offset.unit))
        ax.set_ylabel('Energy ({0})'.format(energy.unit))
        ax.set_title('Energy_offset Array')
        ax.legend()
        image = ax.imshow(self.data.value, extent=extent, **kwargs)
        plt.colorbar(image)
        return ax

    @classmethod
    def read(cls, filename, data_name="data"):
        """Create `EnergyOffsetArray` from  FITS file.

        Parameters
        ----------
        filename : str
            File name
        """
        table = Table.read(filename)
        return cls.from_table(table, data_name)

    @classmethod
    def from_table(cls, table, data_name="data"):
        offset_edges = _make_bin_edges_array(table['THETA_LO'].squeeze(), table['THETA_HI'].squeeze())
        offset_edges = Angle(offset_edges, table['THETA_LO'].unit)
        energy_edges = _make_bin_edges_array(table['ENERG_LO'].squeeze(), table['ENERG_HI'].squeeze())
        energy_edges = EnergyBounds(energy_edges, table['ENERG_LO'].unit)
        data = Quantity(table[data_name].squeeze(), table[data_name].unit)
        return cls(energy_edges, offset_edges, data)

    def write(self, filename, data_name="data", **kwargs):
        """ Write EnergyOffsetArray to FITS file"""
        self.to_table(data_name).write(filename, **kwargs)

    def to_table(self, data_name="data"):
        """Convert `EnergyOffsetArray` to astropy table format.

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

        return table

    def evaluate(self, energy, offset, interpolate_params):
        """
        Interpolate the value of the `EnergyOffsetArray` at a given offset and Energy

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            energy value
        offset : `~astropy.coordinates.Angle`
            offset value
        interpolate_params: dict
            give the options for the RegularGridInterpolator

        Returns
        -------
        Interpolated value
        """
        from scipy.interpolate import RegularGridInterpolator

        energy = Energy(energy).to('TeV')
        offset = Angle(offset).to('deg')

        energy_bin = self.energy.log_centers
        offset_bin = (self.offset.value[:-1] + self.offset.value[1:]) / 2.
        points = (energy_bin, offset_bin)
        interpolator = RegularGridInterpolator(points, self.data.value, **interpolate_params)

        return interpolator([energy.value, offset.value])

    @property
    def bin_volume(self):
        """Per-pixel bin volume.

        TODO: explain with formula and units
        """
        delta_energy = self.energy.bands
        solid_angle = np.pi * (self.offset[1:] ** 2 - self.offset[:-1] ** 2)
        # define grid of deltas (i.e. bin widths for each 3D bin)
        delta_energy, solid_angle = np.meshgrid(delta_energy, solid_angle, indexing='ij')
        bin_volume = delta_energy * (solid_angle).to('sr')

        return bin_volume

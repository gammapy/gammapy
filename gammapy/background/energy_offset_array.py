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


    def __init__(self, energy, offset, data=None):
        self.energy = EnergyBounds(energy)
        self.offset = Angle(offset)
        if data is None:
            self.data = Quantity(np.zeros((len(energy) - 1, len(offset) - 1)), "u")
        else:
            self.data = data

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
            self.data += Quantity(counts, "u")

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
    def read(cls, filename):
        """Create `EnergyOffsetArray` from  FITS file.

        Parameters
        ----------
        filename : str
            File name
        """
        hdu_list = fits.open(filename)
        hdu = hdu_list[1]
        return cls.from_fits(hdu)

    @classmethod
    def from_fits(cls, hdu):
        """Read `EnergyOffsetArray` from a FITS binary table.

        Parameters
        ----------
        hdu : `~astropy.io.fits.BinTableHDU`
            HDU binary table for the EnergyOffsetArray.

        Returns
        -------
        cube : `~gammapy.background.EnergyOffsetArray`
            EnergyOffsetArray object.
        """
        data = hdu.data

        offset_edges = _make_bin_edges_array(data['THETA_LO'].squeeze(), data['THETA_HI'].squeeze())
        energy_edges = _make_bin_edges_array(data['ENERG_LO'].squeeze(), data['ENERG_HI'].squeeze())
        energy_edges = EnergyBounds(energy_edges, 'TeV')

        energy_offset_array = data['EnergyOffsetArray'].squeeze()

        return cls(energy_edges, offset_edges, energy_offset_array)

    def to_table(self):
        """Convert `EnergyOffsetArray` to astropy table format.

        Returns
        -------
        table : `~astropy.table.Table`
            Table containing the EnergyOffsetArray.
        """
        table = Table()
        table['THETA_LO'] = [self.offset[:-1]]
        table['THETA_HI'] = [self.offset[1:]]
        table['ENERG_LO'] = [self.energy[:-1]]
        table['ENERG_HI'] = [self.energy[1:]]
        table['EnergyOffsetArray'] = [Quantity(self.data, " u ")]
        table.meta['name'] = 'TO DEFINE'

        return table

    def to_fits(self):
        """Convert `EnergyOffsetArray` to binary table FITS format.

        Returns
        -------
        tbhdu : `~astropy.io.fits.BinTableHDU`
            Table containing the EnergyOffsetArray.
        """
        return table_to_fits_table(self.to_table())

    def write(self, filename, **kwargs):
        """ Write EnergyOffsetArray to FITS file"""
        self.to_fits().writeto(filename, **kwargs)

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

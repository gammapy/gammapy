# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity
from astropy.table import Table, Column
from astropy.io import fits
from ..utils.fits import table_to_fits_table

__all__ = [
    'EnergyOffsetArray',
]


def _make_bin_edges_array(lo, hi):
    """Make bin edges array from a low values and a high values array.

    TODO: move this function to somewhere else? (i.e. utils?)

    Parameters
    ----------
    lo : `~numpy.ndarray`
        Lower boundaries.
    hi : `~numpy.ndarray`
        Higher boundaries.

    Returns
    -------
    bin_edges : `~numpy.ndarray`
        Array of bin edges as ``[[low], [high]]``.
    """
    return np.append(lo.flatten(), hi.flatten()[-1:])


class EnergyOffsetArray(object):
    """
    Energy offset dependent array.

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
         energy bin vector
    offset : `~astropy.coordinates.Angle`
        offset bin vector
    data : `~numpy.ndarray`
        data array (2D):

    """

    def __init__(self, energy, offset, data=None):
        self.energy = Quantity(energy, 'TeV')
        self.offset = Angle(offset, 'deg')
        if data is None:
            self.data = np.zeros((len(energy) - 1, len(offset) - 1))
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
        # loop over the Lost of object EventList
        for event_list in event_lists:
            # Fill the events
            counts = self._fill_one_event_list(event_list)
            self.data += counts

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

        # stack the offset and energy array
        ev_cube_array = np.vstack([ev_energy, offset]).T

        # fill data cube into histogramdd
        ev_cube_hist, ev_cube_edges = np.histogramdd(ev_cube_array,
                                                     [self.energy.value,
                                                      self.offset.value])
        return ev_cube_hist

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
        ax.imshow(self.data, extent=extent, **kwargs)
        ax.semilogy()
        ax.set_xlabel('Offset ({0})'.format(offset.unit))
        ax.set_ylabel('Energy ({0})'.format(energy.unit))
        ax.set_title('Energy_offset Array')
        ax.legend()
        image = ax.imshow(self.data, extent=extent, **kwargs)
        plt.colorbar(image)
        return ax

    @classmethod
    def read(cls, filename):
        """Create `EnergyOffsetArray` from  fits file.

        Parameters
        ----------
        filename : str
            File name
        """
        hdu_list = fits.open(filename)
        return cls.from_fits(hdu_list)

    @classmethod
    def from_fits(cls, hdu):
        """Read `EnergyOffsetArray` from a fits binary table.

        Parameters
        ----------
        hdu : `~astropy.io.fits.BinTableHDU`
            HDU binary table for the EnergyOffsetArray.

        Returns
        -------
        cube : `~gammapy.background.EnergyOffsetArray`
            EnergyOffsetArray object.
        """

        header = hdu[1].header
        data = hdu[1].data
        # get offset and energy binning
        offset_edges = _make_bin_edges_array(data['THETA_LO'].squeeze(), data['THETA_HI'].squeeze())
        energy_edges = _make_bin_edges_array(data['ENERG_LO'].squeeze(), data['ENERG_HI'].squeeze())

        # get data
        energy_offset_array = data['EnergyOffsetArray'].squeeze()
        return cls(energy_edges, offset_edges, energy_offset_array)

    def to_table(self):
        """Convert `EnergyOffsetArray` to astropy table format.

        Returns
        -------
        table : `~astropy.table.Table`
            Table containing the EnergyOffsetArray.
        """
        # table
        table = Table()
        table['THETA_LO'] = [self.offset[:-1]]
        table['THETA_HI'] = [self.offset[1:]]
        table['ENERG_LO'] = [self.energy[:-1]]
        table['ENERG_HI'] = [self.energy[1:]]
        table['EnergyOffsetArray'] = [Quantity(self.data, " u ")]
        table.meta['name'] = 'TO DEFINE'

        return table

    def to_fits(self):
        """Convert `EnergyOffsetArray` to binary table fits format.

        Returns
        -------
        tbhdu : `~astropy.io.fits.BinTableHDU`
            Table containing the EnergyOffsetArray.
        """
        return table_to_fits_table(self.to_table())

    def write(self, filename, **kwargs):
        """ Write EnergyOffsetArray to fits file"""
        self.to_fits().writeto(filename, **kwargs)

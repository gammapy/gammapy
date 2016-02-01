# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity

__all__ = [
    'EnergyOffsetArray',
]


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
        data array (2D)

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

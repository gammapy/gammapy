# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.units import Quantity, UnitsError
from astropy.coordinates import Angle
from astropy.table import Table
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

    def __init__(self, energy=None, offset=None, data=None):
        self.energy = Quantity(energy, 'TeV')
        self.offset = Angle(offset, 'deg')
        self.data = data

    def fill_events(self, event_list):
        """Fill events histogram.

        This add the counts to the existing value array.

        Parameters
        -------------
        events : list of `~gammapy.data.EventList`
           Python list of event list objects.
            
        
        """

        self.data = None
        # loop over the Lost of object EventList
        for (i, data_set) in enumerate(event_list):
            ev_cube_hist = self.fill_one_event_list(data_set)
            # fill data
            if (i == 0):
                self.data = ev_cube_hist
            else:
                self.data += ev_cube_hist

                
    def fill_one_event_list(self, events):
        """
        histogram the counts of an EventList object in 2D (energy,offset)

        Parameters
        -------------
        events :`~gammapy.data.EventList`
           Event list objects.
                   
        """
        
        try:
            ev_detx = Angle(events['DETX'])
            ev_dety = Angle(events['DETY'])
            ev_energy = Quantity(events['ENERGY'])
        except UnitsError:
            ev_detx = Angle(events['DETX'], 'deg')
            ev_dety = Angle(events['DETY'], 'deg')
            ev_energy = Quantity(events['ENERGY'],
                                 events.meta['EUNIT'])
        # ici calculer offset mis voir si je le calcul avec detx dety ou raddec
        offset = np.sqrt(ev_detx ** 2 + ev_dety ** 2)

        # TODO: filter out possible sources in the data;
        #       for now, the observation table should not contain any
        #       observation at or near an existing source

        # fill events

        # get correct data cube format for histogramdd
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

        # aeff = self.evaluate(offset, energy).T
        extent = [
            offset.value.min(), offset.value.max(),
            energy.value.min(), energy.value.max(),
        ]
        ax.imshow(self.data, extent=extent, **kwargs)
        # ax.set_xlim(offset.value.min(), offset.value.max())
        # ax.set_ylim(energy.value.min(), energy.value.max())

        ax.semilogy()
        ax.set_xlabel('Offset ({0})'.format(offset.unit))
        ax.set_ylabel('Energy ({0})'.format(energy.unit))
        ax.set_title('Energy_offset Array')
        ax.legend()
        plt.colorbar(ax.imshow(self.data, extent=extent, **kwargs))
        return ax

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.units import Quantity, UnitsError
from astropy.coordinates import Angle
from astropy.table import Table
from ..data import DataStore
from ..data import EventListDataset
from astropy.units import Quantity
from astropy.coordinates import Angle

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

    def __init__(self, energy= None, offset=None, data=None):
        self.energy = Quantity(energy, 'TeV')
        self.offset = Angle(offset, 'deg')
        self.data = data



    def fill_events(self,event_list_table):
        """
        from the fill_events method of the Cubebackgroundmodel class in model.py
        Fill events in a 2D energy_offset histograms

        Get data files corresponding to the observation list, histogram
        the counts and fill the corresponding cube containers.

        Parameters
        ----------
        event_list_table : `~astropy.table`
            Astropy table to use for the histogramming that contains three columns: the OBS_ID of the observations to stack, the filetype: "events" and the fits event files for each observation. 
        
        """
        
        self.data = None
        # loop over observations
        for (i,i_ev_file) in enumerate(event_list_table['filename']):
            ev_list_ds = EventListDataset.read(i_ev_file)
            # fill events above energy threshold, correct livetime accordingly
            data_set = ev_list_ds.event_list
           
            # construct histogram (energy, offset) for each event of the list
            # TODO: units are missing in the H.E.S.S. FITS event
            #       lists; this should be solved in the next (prod03)
            #       H.E.S.S. fits production
            # workaround: try to cast units, if it doesn't work, use hard coded
            # ones
            try:
                ev_detx = Angle(data_set['DETX'])
                ev_dety = Angle(data_set['DETY'])
                ev_energy = Quantity(data_set['ENERGY'])
            except UnitsError:
                ev_detx = Angle(data_set['DETX'], 'deg')
                ev_dety = Angle(data_set['DETY'], 'deg')
                ev_energy = Quantity(data_set['ENERGY'],
                                     data_set.meta['EUNIT'])
            #ici calculer offset mis voir si je le calcul avec detx dety ou raddec
            offset=np.sqrt(ev_detx**2 + ev_dety**2)
            ev_cube_table = Table([ev_energy, offset],
                                  names=('ENERGY', 'OFFSET'))

            # TODO: filter out possible sources in the data;
            #       for now, the observation table should not contain any
            #       observation at or near an existing source

            # fill events

            # get correct data cube format for histogramdd
            ev_cube_array = np.vstack([ev_cube_table['ENERGY'],
                                       ev_cube_table['OFFSET']]).T

            # fill data cube into histogramdd
            ev_cube_hist, ev_cube_edges = np.histogramdd(ev_cube_array,
                                                         [self.energy.value,
                                                          self.offset.value])
            #ev_cube_hist = Quantity(ev_cube_hist, '')

            # fill data
            #Ca c'est tres sale donc voir si il y a un autre moyen de le faire.
            self.data += ev_cube_hist
        

    def plot_image(self, ax=None, offset=None, energy=None, **kwargs):
        """Plot Energy_offset Array image (x=offset, y=energy).
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

        #aeff = self.evaluate(offset, energy).T
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

    

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

from ..utils.nddata import NDDataArray, BinnedDataAxis
from astropy.table import Table
import numpy as np

__all__ = [
    'CountsSpectrum',
    #'OnCountsSpectrum',
    #'OffCountsSpectrum',
    'SpectrumObservation',
]

class CountsSpectrum(NDDataArray):
    """Generic counts spectrum

    Parameters
    ----------
    data : `~numpy.array`, list
        Counts
    energy : `~gammapy.utils.energy.EnergyBounds`
        Bin edges of energy axis

    Examples
    --------

    .. plot::
        :include-source:

        from gammapy.spectrum import CountsSpectrum
        import numpy as np
        import astropy.units as u

        ebounds = np.logspace(0,1,10) * u.TeV
        counts = [6,3,8,4,9,5,9,5,5,1]
        spec = CountsSpectrum(data=counts, energy=ebounds)
        spec.plot()
    """
    energy = BinnedDataAxis(interpolation_mode='log')
    """Energy axis"""
    axis_names = ['energy']
    # Use nearest neighbour interpolation for counts
    interp_kwargs = dict(bounds_error=False, method='nearest') 

    @classmethod
    def from_table(cls, table):
        """Read OGIP format table"""
        counts = table['COUNTS'].quantity
        energy_unit = table['BIN_LO'].quantity.unit
        energy = np.append(table['BIN_LO'].data, table['BIN_HI'].data[-1])
        return cls(data=counts, energy=energy*energy_unit)

    def to_table(self):
        """Convert to `~astropy.table.Table`
        
        http://gamma-astro-data-formats.readthedocs.io/en/latest/ogip/index.html
        """
        counts = self.data
        channel = np.arange(1, self.energy.nbins + 1, 1)
        # This is how sherpa save energy information in PHA files
        # https://github.com/sherpa/sherpa/blob/master/sherpa/astro/io/pyfits_backend.py#L643
        bin_lo = self.energy.data[:-1]
        bin_hi = self.energy.data[1:]
        names = ['CHANNEL', 'COUNTS', 'BIN_LO', 'BIN_HI']
        meta = dict(name='SPECTRUM', hduclass='OGIP', hduclas1 = 'SPECTRUM')
        return Table([channel, counts, bin_lo, bin_hi], names=names, meta=meta)

    def fill(self, events):
        """Fill with list of events 
        
        Parameters
        ----------
        events: `~astropy.units.Quantity`, `gammapy.data.EventList`, 
            List of event energies
        """

        if isinstance(events, gammapy.data.EventList):
            events = events.energy

        energy = events.to(self.energy.unit)
        binned_val = np.histogram(energy.value, self.energy.data.value)[0]
        self.data = binned_val 

    @property
    def total_counts(self):
        """Total number of counts
        """
        return self.data.sum()

    def __add__(self, other):
        """Add two counts spectra and returns new instance
        The two spectra need to have the same binning
        """
        if (self.energy.data != other.energy.data).all():
            raise ValueError("Cannot add counts spectra with different binning")
        counts = self.data + other.data
        return CountsSpectrum(data=counts, energy=self.energy)

    def __mul__(self, other):
        """Scale counts by a factor"""
        temp = self.data * other
        return CountsSpectrum(data=temp, energy=self.energy)

    def plot(self, ax=None, energy_unit='TeV', **kwargs):
        """Plot

        kwargs are forwarded to matplotlib.pyplot.hist

        Parameters
        ----------
        ax : `~matplotlib.axis` (optional)
            Axis instance to be used for the plot

        Returns
        -------
        ax: `~matplotlib.axis`
            Axis instance used for the plot
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax
        counts = self.data.value
        enodes = self.energy.nodes.to(energy_unit)
        ebins = self.energy.data.to(energy_unit)
        plt.hist(enodes, bins=ebins, weights=counts, **kwargs)
        plt.xlabel('Energy [{0}]'.format(energy_unit))
        plt.ylabel('Counts')
        plt.semilogx()
        return ax

    def peek(self, figsize=(5, 5)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt
        ax = plt.figure(figsize=figsize)
        self.plot(ax=ax)
        plt.xscale('log')
        plt.show()

    # Todo move to standalone function
    @classmethod
    def get_npred(cls, fit, obs):
        """Get N_pred vector from spectral fit

        Parameters
        ----------
        fit : SpectrumFitResult
            Fitted spectrum
        obs : SpectrumObservationList
            Spectrum observation holding the irfs
        """

        m = fit.to_sherpa_model()

        # Get differential flux at true energy log bin center
        ebounds = EnergyBounds(obs.effective_area.energy.data)
        x = ebounds.log_centers.to('keV')
        diff_flux = Quantity(m(x), 'cm-2 s-1 keV-1')

        # Multiply with bin width = integration
        int_flux = (diff_flux * ebounds.bands).decompose()

        # Apply ARF and RMF to get n_pred
        temp = int_flux * obs.meta.livetime * obs.effective_area.data
        counts = obs.energy_dispersion.pdf_matrix.transpose().dot(temp)

        e_reco = obs.energy_dispersion.reco_energy
        return cls(counts.decompose(), e_reco)


class SpectrumObservation()

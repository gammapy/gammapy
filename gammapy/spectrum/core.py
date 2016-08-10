# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.table import Table
import astropy.units as u
from .. import version
from ..utils.nddata import NDDataArray, BinnedDataAxis
from ..utils.fits import (
    energy_axis_to_ebounds,
    fits_table_to_table,
    ebounds_to_energy_axis,
)
from ..data import EventList

__all__ = [
    'CountsSpectrum',
    'PHACountsSpectrum',
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

        ebounds = np.logspace(0,1,11) * u.TeV
        counts = np.arange(10) * u.ct
        spec = CountsSpectrum(data=counts, energy=ebounds)
        spec.plot(show_poisson_errors=True)
    """
    energy = BinnedDataAxis(interpolation_mode='log')
    """Energy axis"""
    axis_names = ['energy']
    # Use nearest neighbour interpolation for counts
    interp_kwargs = dict(bounds_error=False, method='nearest')

    @classmethod
    def from_hdulist(cls, hdulist):
        """Read OGIP format hdulist"""
        counts_table = fits_table_to_table(hdulist[1])
        counts = counts_table['COUNTS'] * u.ct
        ebounds = ebounds_to_energy_axis(hdulist[2])
        return cls(data=counts, energy=ebounds)

    def to_table(self):
        """Convert to `~astropy.table.Table`
        
        http://gamma-astro-data-formats.readthedocs.io/en/latest/ogip/index.html
        """
        channel = np.arange(self.energy.nbins, dtype=np.int16)
        counts = np.array(self.data.value, dtype=np.int32)

        names = ['CHANNEL', 'COUNTS']
        meta = dict()
        return Table([channel, counts], names=names, meta=meta)

    def to_hdulist(self):
        """Convert to `~astropy.io.fits.HDUList`

        This adds an ``EBOUNDS`` extension to the ``BinTableHDU`` produced by 
        ``to_table``, in order to store the energy axis
        """
        hdulist = super(CountsSpectrum, self).to_hdulist()
        ebounds = energy_axis_to_ebounds(self.energy)
        hdulist.append(ebounds)
        return hdulist

    def fill(self, events):
        """Fill with list of events 
        
        Parameters
        ----------
        events: `~astropy.units.Quantity`, `gammapy.data.EventList`, 
            List of event energies
        """

        if isinstance(events, EventList):
            events = events.energy

        energy = events.to(self.energy.unit)
        binned_val = np.histogram(energy.value, self.energy.data.value)[0]
        self.data = binned_val * u.ct

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

    def __sub__(self, other):
        """Subtract two CountsSpectra"""
        return self.__add__(other.__mul__(-1))

    def plot(self, ax=None, energy_unit='TeV', show_poisson_errors=False,
             show_energy=None, **kwargs):
        """Plot as datapoint

        kwargs are forwarded to `~matplotlib.pyplot.errorbar`

        Parameters
        ----------
        ax : `~matplotlib.axis` (optional)
            Axis instance to be used for the plot
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        show_poisson_errors : bool, optional
            Show poisson errors on the plot
        show_energy : `~astropy.units.Quantity`, optional
            Show energy, e.g. threshold, as vertical line

        Returns
        -------
        ax: `~matplotlib.axis`
            Axis instance used for the plot
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax
        counts = self.data.value
        x = self.energy.nodes.to(energy_unit).value
        bounds = self.energy.data.to(energy_unit).value
        xerr = [x - bounds[:-1], bounds[1:] - x]
        yerr = np.sqrt(counts) if show_poisson_errors else 0
        kwargs.setdefault('fmt', '')
        ax.errorbar(x, counts, xerr=xerr, yerr=yerr, **kwargs)
        if show_energy is not None:
            ener_val = u.Quantity(show_energy).to(energy_unit).value
            ax.vlines(ener_val, 0, 1.1 * max(self.data.value),
                      linestyles='dashed')
        ax.set_xlabel('Energy [{0}]'.format(energy_unit))
        ax.set_ylabel('Counts')
        ax.set_xscale('log')
        ax.set_ylim(0, 1.2 * max(self.data.value))
        return ax

    def plot_hist(self, ax=None, energy_unit='TeV', show_energy=None, **kwargs):
        """Plot as histogram
        
        kwargs are forwarded to `~matplotlib.pyplot.hist`

        Parameters
        ----------
        ax : `~matplotlib.axis` (optional)
            Axis instance to be used for the plot
        energy_unit : str, `~astropy.units.Unit`, optional
            Unit of the energy axis
        show_energy : `~astropy.units.Quantity`, optional
            Show energy, e.g. threshold, as vertical line
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax
        kwargs.setdefault('lw', 2)
        kwargs.setdefault('histtype', 'step')
        weights = self.data.value
        bins = self.energy.data.to(energy_unit).value[:-1]
        x = self.energy.nodes.to(energy_unit).value
        ax.hist(x, bins=bins, weights=weights, **kwargs)
        if show_energy is not None:
            ener_val = u.Quantity(show_energy).to(energy_unit).value
            ax.vlines(ener_val, 0, 1.1 * max(self.data.value),
                      linestyles='dashed')
        ax.set_xlabel('Energy [{0}]'.format(energy_unit))
        ax.set_ylabel('Counts')
        ax.set_xscale('log')
        return ax

    def peek(self, figsize=(5, 10)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.plot_hist(ax=ax)
        return ax


class PHACountsSpectrum(CountsSpectrum):
    """OGIP PHA equivalent
    
    The ``bkg`` flag controls wheater the PHA counts spectrum represents a
    background estimate or not (this slightly affectes the FITS header
    information when writing to disk).

    Parameters
    ----------
    data : `~numpy.array`, list
        Counts
    energy : `~astropy.units.Quantity`
        Bin edges of energy axis
    obs_id : int
        Unique identifier
    livetime : `~astropy.units.Quantity`
        Observation live time
    backscal : float
        Scaling factor
    lo_threshold : `~astropy.units.Quantity`
        Low energy threshold
    hi_threshold : `~astropy.units.Quantity`
        High energy threshold
    is_bkg : bool, optional
        Background or soure spectrum, default: False
    telescope : str, optional
        Mission name
    instrument : str, optional
        Instrument, detector
    creator : str, optional
        Software used to produce the PHA file
    tstart :  `~astropy.time.Time`, optional
        Time start MJD
    tstop :  `~astropy.time.Time`, optional
        Time stop MJD
    muoneff : float, optional
        Muon efficiency
    zen_pnt : `~astropy.coordinates.Angle`, optional
        Zenith Angle
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('is_bkg', False)
        super(CountsSpectrum, self).__init__(**kwargs)
        if not self.is_bkg:
            self.phafile = 'pha_obs{}.fits'.format(self.obs_id)
            self.arffile = self.phafile.replace('pha', 'arf')
            self.rmffile = self.phafile.replace('pha', 'rmf')
            self.bkgfile = self.phafile.replace('pha', 'bkg')

    @property
    def quality(self):
        """Mask bins outside energy thresholds (1 = bad, 0 = good)"""
        flag = np.zeros(self.energy.nbins, dtype=np.int16)
        idx = np.where((self.energy.data[:-1] < self.lo_threshold) |
                       (self.energy.data[1:] > self.hi_threshold))
        flag[idx] = 1
        return flag

    def to_table(self):
        """Write"""
        table = super(PHACountsSpectrum, self).to_table()

        table['QUALITY'] = self.quality

        meta = dict(name='SPECTRUM',
                    hduclass='OGIP',
                    hduclas1='SPECTRUM',
                    obs_id=self.obs_id,
                    exposure=self.livetime.to('s').value,
                    backscal=float(self.backscal),
                    corrscal='',
                    areascal=1,
                    chantype='PHA',
                    detchans=self.energy.nbins,
                    filter='None',
                    corrfile='',
                    poisserr=True,
                    telescop=getattr(self, 'telescope', 'HESS'),
                    instrume=getattr(self, 'instrument', 'CT1234'),
                    creator=getattr(self, 'creator', 'Gammapy {}'.format(
                        version.version)),
                    hduclas3='COUNT',
                    hduclas4='TYPE:1',
                    lo_thres=self.lo_threshold.to("TeV").value,
                    hi_thres=self.hi_threshold.to("TeV").value,
                    )
        if not self.is_bkg:
            if self.rmffile is not None:
                meta.update(respfile=self.rmffile)

            meta.update(backfile=self.bkgfile,
                        ancrfile=self.arffile,
                        hduclas2='TOTAL', )
        else:
            meta.update(hduclas2='BKG', )

        # Add general optional keywords if the member exists rather than default value. LBYL approach
        if hasattr(self, 'tstart'):
            meta.update(tstart=self.tstart.mjd)

        if hasattr(self, 'tstop'):
            meta.update(tstop=self.tstop.mjd)

        if hasattr(self, 'muoneff'):
            meta.update(muoneff=self.muoneff)

        if hasattr(self, 'zen_pnt'):
            meta.update(zen_pnt=self.zen_pnt.to('deg').value)

        table.meta = meta
        return table

    @classmethod
    def from_hdulist(cls, hdulist):
        """Read"""
        counts_table = fits_table_to_table(hdulist[1])
        counts = counts_table['COUNTS'] * u.ct
        ebounds = ebounds_to_energy_axis(hdulist[2])
        meta = dict(
            obs_id=hdulist[1].header['OBS_ID'],
            livetime=hdulist[1].header['EXPOSURE'] * u.s,
            backscal=hdulist[1].header['BACKSCAL'],
            lo_threshold=hdulist[1].header['LO_THRES'] * u.TeV,
            hi_threshold=hdulist[1].header['HI_THRES'] * u.TeV,
        )
        if hdulist[1].header['HDUCLAS2'] == 'BKG':
            meta.update(is_bkg=True)
        return cls(energy=ebounds, data=counts, **meta)
    
    def to_sherpa(self, name):
        """Return `~sherpa.astro.data.DataPHA`
        
        Parameters
        ----------
        name : str
            Instance name
        """
        from sherpa.utils import SherpaFloat
        from sherpa.astro.data import DataPHA

        table = self.to_table()
        kwargs = dict(
            name = name, 
            channel = (table['CHANNEL'].data + 1).astype(SherpaFloat), 
            counts = table['COUNTS'].data.astype(SherpaFloat), 
            quality = table['QUALITY'].data,
            exposure = self.livetime.to('s').value,
            backscal = self.backscal,
            areascal = 1.,
            syserror = None,
            staterror = None,
            grouping = None,
        )

        return DataPHA(**kwargs)
        

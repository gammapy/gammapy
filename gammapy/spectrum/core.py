# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from copy import deepcopy
import numpy as np
from astropy.table import Table
from astropy.io import fits
import astropy.units as u
from .. import version
from ..utils.nddata import NDDataArray, BinnedDataAxis
from ..utils.scripts import make_path
from ..utils.fits import (
    energy_axis_to_ebounds,
    fits_table_to_table,
    ebounds_to_energy_axis,
    table_to_fits_table,
)
from ..data import EventList

__all__ = [
    'CountsSpectrum',
    'PHACountsSpectrum',
]


class CountsSpectrum(object):
    """Generic counts spectrum

    Parameters
    ----------
    energy : `~gammapy.utils.energy.EnergyBounds`
        Bin edges of energy axis
    data : `~astropy.units.Quantity`, array-like
        Counts

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
    default_interp_kwargs = dict(bounds_error=False, method='nearest')
    """Default interpolation kwargs"""

    def __init__(self, energy, data=None, interp_kwargs=None):
        axes = [BinnedDataAxis(energy, interpolation_mode='log', name='energy')]
        # Set data unit to counts for coherence
        if data is not None:
            if isinstance(data, u.Quantity):
                if data.unit.is_equivalent('ct'):
                    pass
                elif data.unit.is_equivalent(u.Unit('')):
                    data = data.value
                else:
                    raise ValueError('Invalid data unit {}'.format(data.unit))
            data = u.Quantity(data, 'ct')

        if interp_kwargs is None:
            interp_kwargs = self.default_interp_kwargs
        self.data = NDDataArray(axes=axes, data=data, interp_kwargs=interp_kwargs)

    @classmethod
    def from_hdulist(cls, hdulist, hdu1='COUNTS', hdu2='EBOUNDS'):
        """Read OGIP format hdulist"""
        counts_table = fits_table_to_table(hdulist[hdu1])
        counts = counts_table['COUNTS'].data
        ebounds = ebounds_to_energy_axis(hdulist[hdu2])
        return cls(data=counts, energy=ebounds)

    @classmethod
    def read(cls, filename, hdu1='COUNTS', hdu2='EBOUNDS', **kwargs):
        filename = make_path(filename)
        hdulist = fits.open(str(filename), **kwargs)
        try:
            return cls.from_hdulist(hdulist, hdu1=hdu1, hdu2=hdu2)
        except KeyError:
            msg = 'File {} does not contain HDUs "{}"'.format(
                filename, [hdu1, hdu2])
            msg += '\n Available {}'.format([_.name for _ in hdulist])
            raise ValueError(msg)

    def to_table(self):
        """Convert to `~astropy.table.Table`

        http://gamma-astro-data-formats.readthedocs.io/en/latest/ogip/index.html
        """
        channel = np.arange(self.data.axis('energy').nbins, dtype=np.int16)
        counts = np.array(self.data.data.value, dtype=np.int32)

        names = ['CHANNEL', 'COUNTS']
        meta = dict(name='COUNTS')
        return Table([channel, counts], names=names, meta=meta)

    def to_hdulist(self):
        """Convert to `~astropy.io.fits.HDUList`

        This adds an ``EBOUNDS`` extension to the ``BinTableHDU`` produced by
        ``to_table``, in order to store the energy axis
        """
        hdu = table_to_fits_table(self.to_table())
        prim_hdu = fits.PrimaryHDU()
        ebounds = energy_axis_to_ebounds(self.data.axis('energy').data)
        return fits.HDUList([prim_hdu, hdu, ebounds])

    def write(self, filename, **kwargs):
        filename = make_path(filename)
        self.to_hdulist().writeto(str(filename), **kwargs)

    def fill(self, events):
        """Fill with list of events

        Parameters
        ----------
        events: `~astropy.units.Quantity`, `gammapy.data.EventList`,
            List of event energies
        """
        if isinstance(events, EventList):
            events = events.energy

        energy = events.to(self.data.axis('energy').unit)
        binned_val = np.histogram(energy.value, self.data.axis('energy').data.value)[0]
        self.data.data = binned_val * u.ct

    @property
    def total_counts(self):
        """Total number of counts
        """
        return self.data.data.sum()

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
        counts = self.data.data.value
        x = self.data.axis('energy').nodes.to(energy_unit).value
        bounds = self.data.axis('energy').data.to(energy_unit).value
        xerr = [x - bounds[:-1], bounds[1:] - x]
        yerr = np.sqrt(counts) if show_poisson_errors else 0
        kwargs.setdefault('fmt', '')
        ax.errorbar(x, counts, xerr=xerr, yerr=yerr, **kwargs)
        if show_energy is not None:
            ener_val = u.Quantity(show_energy).to(energy_unit).value
            ax.vlines(ener_val, 0, 1.1 * max(self.data.data.value),
                      linestyles='dashed')
        ax.set_xlabel('Energy [{0}]'.format(energy_unit))
        ax.set_ylabel('Counts')
        ax.set_xscale('log')
        ax.set_ylim(0, 1.2 * max(self.data.data.value))
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
        weights = self.data.data.value
        bins = self.data.axis('energy').data.to(energy_unit).value[:-1]
        x = self.data.axis('energy').nodes.to(energy_unit).value
        ax.hist(x, bins=bins, weights=weights, **kwargs)
        if show_energy is not None:
            ener_val = u.Quantity(show_energy).to(energy_unit).value
            ax.vlines(ener_val, 0, 1.1 * max(self.data.data.value),
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

    def copy(self):
        """A deep copy of self.
        """
        return deepcopy(self)

    def spectral_index(self, energy, dz=1e-3):
        """Power law spectral index (`numpy.array`).

        A forward finite difference method with step ``dz`` is used along
        the ``z = log10(energy)`` axis.

        Parameters
        ----------
        lon : `~astropy.coordinates.Angle`
            Longitude
        lat : `~astropy.coordinates.Angle`
            Latitude
        energy : `~astropy.units.Quantity`
            Energy
        """
        raise NotImplementedError
        # Compute flux at `z = log(energy)`
        pix_coord = self.world2pix(lon, lat, energy, combine=True)
        flux1 = self._interpolate(pix_coord)

        # Compute flux at `z + dz`
        pix_coord[:, 0] += dz
        # pixel_coordinates += np.zeros(pixel_coordinates.shape)
        flux2 = self._interpolate(pix_coord)

        # Power-law spectral index through these two flux points
        # d_log_flux = np.log(flux2 / flux1)
        # spectral_index = d_log_flux / dz
        energy1 = energy
        energy2 = (1. + dz) * energy
        spectral_index = powerlaw.g_from_points(energy1, energy2, flux1, flux2)

        return spectral_index

    def rebin(self, parameter):
        """Rebin

        Parameters
        ----------
        parameter, int
            Number of bins to merge

        Returns
        -------
        rebinned_spectrum : `~gammapy.spectrum.CountsSpectrum`
            Rebinned spectrum
        """
        if len(self.data.data) % parameter != 0:
            raise ValueError("Invalid rebin parameter: {}, nbins: {}".format(
                parameter, len(self.data.data)))

        # Copy to keep attributes
        retval = self.copy()
        energy = retval.data.axis('energy')
        energy.data = energy.data[0::parameter]
        split_indices = np.arange(parameter, len(retval.data.data), parameter)
        counts_grp = np.split(retval.data.data, split_indices)
        counts_rebinned = np.sum(counts_grp, axis=1)
        retval.data.data = counts_rebinned * u.ct

        return retval


class PHACountsSpectrum(CountsSpectrum):
    """OGIP PHA equivalent

    The ``bkg`` flag controls wheater the PHA counts spectrum represents a
    background estimate or not (this slightly affectes the FITS header
    information when writing to disk).

    TODO: Bundle optional parameters as meta

    Parameters
    ----------
    data : `~numpy.array`, list
        Counts
    energy : `~astropy.units.Quantity`
        Bin edges of energy axis
    obs_id : int, optional
    Unique identifier, default: 0
    quality : int, array-lik
        Mask bins in safe energy range (1 = bad, 0 = good)
    is_bkg : bool, optional
        Background or soure spectrum, default: False
    livetime : `~astropy.units.Quantity`
        Observation live time
    backscal : float, array-like
        Scaling factor for each bin
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

    def __init__(self, energy, data, obs_id=0, quality=None, is_bkg=False,
                 **kwargs):
        super(PHACountsSpectrum, self).__init__(energy, data)
        self.obs_id=obs_id
        if quality is None:
            quality = np.zeros(self.data.axis('energy').nbins, dtype=int)
        self.quality = quality
        self.is_bkg = is_bkg

    @property
    def phafile(self):
        """PHA file associated with the observations"""
        return 'pha_obs{}.fits'.format(self.obs_id)

    @property
    def arffile(self):
        """ARF associated with the observations"""
        return self.phafile.replace('pha', 'arf')

    @property
    def rmffile(self):
        """RMF associated with the observations"""
        return self.phafile.replace('pha', 'rmf')

    @property
    def bkgfile(self):
        """Background PHA files associated with the observations"""
        return self.phafile.replace('pha', 'bkg')

    @property
    def bins_in_safe_range(self):
        """Indices of bins within the energy thresholds"""
        idx = np.where(np.array(self.quality) == 0)[0]
        return idx

    @property
    def counts_in_safe_range(self):
        """Counts with bins outside safe range set to 0"""
        data = self.data.data.copy()
        data[np.nonzero(self.quality)] = 0
        return data

    @property
    def lo_threshold(self):
        """Low energy threshold of the observation (lower bin edge)"""
        idx = self.bins_in_safe_range[0]
        return self.data.axis('energy').data[idx]

    @lo_threshold.setter
    def lo_threshold(self, thres):
        idx = np.where(self.data.axis('energy').data < thres)[0]
        self.quality[idx] = 1

    @property
    def hi_threshold(self):
        """High energy threshold of the observation (upper bin edge)"""
        idx = self.bins_in_safe_range[-1]
        return self.data.axis('energy').data[idx + 1]

    @hi_threshold.setter
    def hi_threshold(self, thres):
        idx = np.where(self.data.axis('energy').data[:-1] > thres)[0]
        if len(idx) != 0:
            idx = np.insert(idx, 0, idx[0] - 1)
        self.quality[idx] = 1

    def rebin(self, parameter):
        """Rebin

        see `~gammapy.spectrum.CountsSpectrum`, this function treats the
        quality vector correctly
        """
        retval = super(PHACountsSpectrum, self).rebin(parameter)
        split_indices = np.arange(parameter, len(self.data.data), parameter)
        quality_grp = np.split(retval.quality, split_indices)
        quality_summed = np.sum(quality_grp, axis=1)
        # Exclude groups where not all bins are within the safe threshold
        condition = (quality_summed == parameter)
        quality_rebinned = np.where(condition,
                                    np.ones(len(retval.data.data)),
                                    np.zeros(len(retval.data.data)))
        retval.quality = np.array(quality_rebinned, dtype=int)
        return retval

    @property
    def _backscal_array(self):
        """Helper function to always return backscal as an array"""
        if np.isscalar(self.backscal):
            return np.ones(self.data.axis('energy').nbins) * self.backscal
        else:
            return self.backscal

    def to_table(self):
        """Write"""
        table = super(PHACountsSpectrum, self).to_table()

        table['QUALITY'] = self.quality
        table['BACKSCAL'] = self._backscal_array

        meta = dict(name='SPECTRUM',
                    hduclass='OGIP',
                    hduclas1='SPECTRUM',
                    obs_id=self.obs_id,
                    exposure=self.livetime.to('s').value,
                    corrscal='',
                    areascal=1,
                    chantype='PHA',
                    detchans=self.data.axis('energy').nbins,
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
    def from_hdulist(cls, hdulist, hdu1='SPECTRUM', hdu2='EBOUNDS'):
        """Read"""
        counts_table = fits_table_to_table(hdulist[hdu1])
        kwargs = dict(
            data=counts_table['COUNTS'] * u.ct,
            energy=ebounds_to_energy_axis(hdulist[hdu2]),
            backscal=counts_table['BACKSCAL'].data,
            quality=counts_table['QUALITY'].data,
            obs_id=hdulist[1].header['OBS_ID'],
            livetime=hdulist[1].header['EXPOSURE'] * u.s,
        )
        if hdulist[1].header['HDUCLAS2'] == 'BKG':
            kwargs.update(is_bkg=True)
        return cls(**kwargs)

    @classmethod
    def read(cls, filename, hdu1='SPECTRUM', hdu2='EBOUNDS', **kwargs):
        filename = make_path(filename)
        hdulist = fits.open(str(filename), **kwargs)
        try:
            return cls.from_hdulist(hdulist, hdu1=hdu1, hdu2=hdu2)
        except KeyError:
            msg = 'File {} does not contain HDUs "{}"'.format(
                filename, [hdu1, hdu2])
            msg += '\n Available {}'.format([_.name for _ in hdulist])
            raise ValueError(msg)

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

        # Workaround to avoid https://github.com/sherpa/sherpa/issues/248
        # TODO: Remove
        if np.isscalar(self.backscal):
            backscal = self.backscal
        else:
            backscal = self.backscal.copy()
            if np.allclose(backscal.mean(), backscal):
                backscal = backscal[0]

        kwargs = dict(
            name=name,
            channel=(table['CHANNEL'].data + 1).astype(SherpaFloat),
            counts=table['COUNTS'].data.astype(SherpaFloat),
            quality=table['QUALITY'].data,
            exposure=self.livetime.to('s').value,
            backscal=backscal,
            areascal=1.,
            syserror=None,
            staterror=None,
            grouping=None,
        )

        return DataPHA(**kwargs)


class PHACountsSpectrumList(list):
    """List of `~gammapy.spectrum.PHACountsSpectrum

    All spectra must have the same energy binning. This represent the PHA type
    II data format. See
    https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007/node8.html
    """

    def write(self, outdir, **kwargs):
        """Write to file"""
        outdir = make_path(outdir)
        self.to_hdulist().writeto(str(outdir), **kwargs)

    def to_hdulist(self):
        """Create `~astropy.fits.HDUList`"""
        prim_hdu = fits.PrimaryHDU()
        hdu = table_to_fits_table(self.to_table())
        ebounds = energy_axis_to_ebounds(self[0].energy)
        return fits.HDUList([prim_hdu, hdu, ebounds])

    def to_table(self):
        """Create `~astropy.table.Table`
        """
        is_bkg = self[0].is_bkg
        nbins = self[0].energy.nbins
        spec_num = np.empty([len(self), 1], dtype=np.int16)
        channel = np.empty([len(self), nbins], dtype=np.int16)
        counts = np.empty([len(self), nbins], dtype=np.int32)
        quality = np.empty([len(self), nbins], dtype=np.int32)
        backscal = np.empty([len(self), nbins], dtype=np.int32)
        backfile = list()
        for idx, pha in enumerate(self):
            t = pha.to_table()
            spec_num[idx] = pha.obs_id
            channel[idx] = t['CHANNEL'].data
            counts[idx] = t['COUNTS'].data
            quality[idx] = t['QUALITY'].data
            backscal[idx] = t['BACKSCAL'].data
            backfile.append('bkg.fits[{}]'.format(idx))

        names = ['SPEC_NUM', 'CHANNEL', 'COUNTS', 'QUALITY', 'BACKSCAL']
        meta = self[0].to_table().meta
        meta['hduclas4'] = 'TYPE:II'
        meta['ancrfile'] = 'arf.fits'
        meta['respfile'] = 'rmf.fits'
        meta.pop('obs_id')

        table = Table([spec_num, channel, counts, quality, backscal],
                      names=names,
                      meta=meta)

        if not is_bkg:
            table.meta.pop('backfile')
            table['BACKFILE'] = backfile

        return table

    @classmethod
    def read(cls, filename):
        """Read from file"""
        filename = make_path(filename)
        hdulist = fits.open(str(filename))
        return cls.from_hdulist(hdulist)

    @classmethod
    def from_hdulist(cls, hdulist):
        """Create from `~astropy.fits.HDUList`"""
        counts_table = fits_table_to_table(hdulist[1])
        kwargs = dict(
            energy=ebounds_to_energy_axis(hdulist[2]),
            livetime=hdulist[1].header['EXPOSURE'] * u.s,
        )
        if hdulist[1].header['HDUCLAS2'] == 'BKG':
            kwargs.update(is_bkg=True)

        counts_table = fits_table_to_table(hdulist[1])
        speclist = cls()
        for row in counts_table:
            kwargs['data'] = row['COUNTS'] * u.ct
            kwargs['backscal'] = row['BACKSCAL']
            kwargs['quality'] = row['QUALITY']
            kwargs['obs_id'] = row['SPEC_NUM']
            speclist.append(PHACountsSpectrum(**kwargs))

        return speclist

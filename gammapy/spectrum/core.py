# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.table import Table
from ..utils.nddata import NDDataArray, BinnedDataAxis
from ..utils.scripts import make_path
from ..data import EventList
from ..extern.pathlib import Path
import astropy.units as u
import numpy as np

__all__ = [
    'CountsSpectrum',
    'PHACountsSpectrum',
    'SpectrumObservation',
    'SpectrumObservationList',
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
        counts = [6,3,8,4,9,5,9,5,5,1] * u.ct
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
        # energy_unit = table['BIN_LO'].quantity.unit
        # energy = np.append(table['BIN_LO'].data, table['BIN_HI'].data[-1])
        # return cls(data=counts, energy=energy*energy_unit)
        return cls(data=counts)

    def to_table(self):
        """Convert to `~astropy.table.Table`
        
        http://gamma-astro-data-formats.readthedocs.io/en/latest/ogip/index.html
        """
        channel = np.arange(self.energy.nbins, dtype=np.int16)
        counts = np.array(self.data.value, dtype=np.int32)
        # This is how sherpa save energy information in PHA files
        # https://github.com/sherpa/sherpa/blob/master/sherpa/astro/io/pyfits_backend.py#L643
        #bin_lo = self.energy.data[:-1]
        #bin_hi = self.energy.data[1:]
        #names = ['CHANNEL', 'COUNTS', 'BIN_LO', 'BIN_HI']
        #meta = dict()
        #return Table([channel, counts, bin_lo, bin_hi], names=names, meta=meta)
        names = ['CHANNEL', 'COUNTS']
        meta = dict()
        return Table([channel, counts], names=names, meta=meta)
        
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

    # TODO : move to standalone function and fix
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
        energy = obs.aeff.energy
        x = energy.nodes.to('keV')
        diff_flux = m(x) * u.Unit('cm-2 s-1 keV-1')

        # Multiply with bin width = integration
        bands = energy.data[:-1] - energy.data[1:]
        int_flux = (diff_flux * bands).decompose()

        # Apply ARF and RMF to get n_pred
        temp = int_flux * obs.on_vector.exposure * obs.aeff.data
        counts = obs.edisp.pdf_matrix.transpose().dot(temp)

        e_reco = obs.edisp.reco_energy
        return cls(data=counts.decompose(), energy=e_reco)


class PHACountsSpectrum(CountsSpectrum):
    """OGIP PHA equivalent
    
    The ``bkg`` flag controls wheater the PHA counts spectrum represents a
    background estimate or not (this slightly affectes the FITS header
    information when writing to disk).

    Parameters
    ----------
    data : `~numpy.array`, list
        Counts
    energy : `~gammapy.utils.energy.EnergyBounds`
        Bin edges of energy axis
    obs_id : int
        Unique identifier
    exposure : `~astropy.units.Quantity`
        Observation live time
    backscal : float
        Scaling factor
    is_bkg : bool, optional
        Background or soure spectrum, default: False
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('is_bkg', False)
        super(CountsSpectrum, self).__init__(**kwargs)
        if not self.is_bkg:
            self.phafile = 'pha_obs{}.fits'.format(self.obs_id)
            self.arffile = self.phafile.replace('pha', 'arf')
            self.rmffile = self.phafile.replace('pha', 'rmf')
            self.bkgfile = self.phafile.replace('pha', 'bkg')

    def to_table(self):
        """Write"""
        table = super(PHACountsSpectrum, self).to_table()
        # Remove BIN_LO and BIN_HI columns for now
        # see https://github.com/gammapy/gammapy/issues/561
        table.remove_columns(['BIN_LO', 'BIN_HI'])
        meta = dict(name='SPECTRUM',
                    hduclass='OGIP',
                    hduclas1='SPECTRUM',
                    obs_id=self.obs_id,
                    exposure=self.exposure.to('s').value,
                    backscal=float(self.backscal),
                    corrscal='',
                    areascal=1,
                    chantype='PHA',
                    detchans=self.energy.nbins,
                    filter=None,
                    corrfile='',
                    poisserr=True,
                    )
        if not self.is_bkg:
            meta.update(backfile=self.bkgfile,
                        respfile=self.rmffile,
                        ancrfile=self.arffile,
                        lo_threshold=self.lo_threshold.to("TeV").value,
                        hi_threshold=self.hi_threshold.to("TeV").value,
                        hduclas2='TOTAL', )
        else:
            meta.update(hduclas2='BKG', )

        table.meta = meta
        return table

    @classmethod
    def from_table(cls, table):
        """Read"""
        # This will fail since BIN_LO and BIN_HI are not present
        # spec = CountsSpectrum.from_table(table)
        counts = table['COUNTS'].quantity
        # FIXME : set energy to CHANNELS * u.eV, WRONG 
        energy = np.append([0], table['CHANNEL'].data) * u.eV
        meta = dict(
            obs_id=table.meta['OBS_ID'],
            exposure=table.meta['EXPOSURE'] * u.s,
            backscal=table.meta['BACKSCAL']
        )
        if table.meta["HDUCLAS2"] == "TOTAL":
            meta.update(lo_threshold=table.meta['lo_threshold'] * u.TeV,
                        hi_threshold=table.meta['hi_threshold'] * u.TeV)
        return cls(energy=energy, data=counts, **meta)


class SpectrumObservation(object):
    """1D spectral analysis storage class
    
    This container holds the ingredients for 1D region based spectral analysis
    TODO: describe PHA, ARF, etc.

    Meta data is stored in the ``on_vector`` attribute. This reflects the OGIP
    convention.

    Parameters
    ----------
    on_vector : `~gammapy.spectrum.PHACountsSpectrum`
        On vector
    off_vector : `~gammapy.spectrum.PHACountsSpectrum`
        Off vector
    aeff : `~gammapy.irg.EffectiveAreaTable`
        Effective Area
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion matrix
    """

    def __init__(self, on_vector, off_vector, aeff, edisp):
        self.on_vector = on_vector
        self.off_vector = off_vector
        self.aeff = aeff
        self.edisp = edisp

    @property
    def obs_id(self):
        return self.on_vector.obs_id

    @property
    def lo_threshold(self):
        return self.on_vector.lo_threshold

    @property
    def hi_threshold(self):
        return self.on_vector.hi_threshold

    @property
    def phafile(self):
        return self._phafile

    @classmethod
    def read(cls, phafile):
        """Read `~gammapy.spectrum.SpectrumObservation` from OGIP files.

        BKG file, ARF, and RMF must be set in the PHA header and be present in
        the same folder.

        Parameters
        ----------
        phafile : str
            OGIP PHA file to read
        """
        # Put here due to circular imports issues
        from ..irf import EnergyDispersion, EffectiveAreaTable

        f = make_path(phafile)
        base = f.parent
        on_vector = PHACountsSpectrum.read(f)
        rmf, arf, bkg = on_vector.rmffile, on_vector.arffile, on_vector.bkgfile
        energy_dispersion = EnergyDispersion.read(str(base / rmf))
        effective_area = EffectiveAreaTable.read(str(base / arf))
        off_vector = PHACountsSpectrum.read(str(base / bkg))

        # This is needed for know since when passing a SpectrumObservation to
        # the fitting class actually the PHA file is loaded again
        # TODO : remove one spectrumfit is updated
        retval =  cls(on_vector, off_vector, effective_area, energy_dispersion)
        retval._phafile = phafile
        return retval

    def write(self, outdir=None, overwrite=True):
        """Write OGIP files

        The files are meant to be used in Sherpa. The units are therefore
        hardcoded to 'keV' and 'cm2'.

        Parameters
        ----------
        outdir : `~gammapy.extern.pathlib.Path`
            output directory, default: pwd
        overwrite : bool
            Overwrite
        """

        outdir = Path(outdir) or Path.cwd()
        outdir.mkdir(exist_ok=True, parents=True)

        phafile = self.on_vector.phafile
        bkgfile = self.on_vector.bkgfile
        arffile = self.on_vector.arffile
        rmffile = self.on_vector.rmffile
        self._phafile = outdir / phafile
        # Write in keV and cm2
        self.on_vector.energy.data = self.on_vector.energy.data.to('keV')
        self.on_vector.write(outdir / phafile, overwrite=overwrite)

        self.off_vector.energy.data = self.off_vector.energy.data.to('keV')
        self.off_vector.write(outdir / bkgfile, overwrite=overwrite)

        self.aeff.data = self.aeff.data.to('cm2')
        self.aeff.energy.data = self.aeff.energy.data.to('keV')
        self.aeff.write(outdir / arffile, overwrite=overwrite)

        self.edisp.write(str(outdir / rmffile), energy_unit='keV',
                         clobber=overwrite)

    @property
    def excess_vector(self):
        """Excess vector
        Excess = n_on - alpha * n_off
        """
        return self.on_vector + self.off_vector * self.alpha * -1

    @property
    def spectrum_stats(self):
        """`~gammapy.spectrum.results.SpectrumStats`
        """
        n_on = self.on_vector.total_counts
        n_off = self.off_vector.total_counts
        val = dict()
        val['n_on'] = n_on
        val['n_off'] = n_off
        val['alpha'] = self.alpha
        val['excess'] = float(n_on) - float(n_off) * self.alpha
        val['energy_range'] = self.meta.energy_range
        return SpectrumStats(**val)

    def apply_energy_cut(self, energy_range=None, method='binned'):
        """Restrict to a given energy range

        If no energy range is given, it will be extracted from the PHA header.
        Tow methods are available . Unbinned method: The new counts vectors are
        created from the list of on and off events. Therefore this list must be
        saved in the meta info. Binned method: The counts vectors are taken as
        a basis for the energy range restriction. Only bins that are entirely
        contained in the desired energy range are copied.

        Parameters
        ----------
        energy_range : `~gammapy.utils.energy.EnergyBounds`, optional
            Desired energy range
        method : str {'unbinned', 'binned'}
            Use unbinned on list / binned on vector

        Returns
        -------
        obs : `~gammapy.spectrum.spectrum_extraction.SpectrumObservation`
            Spectrum observation in desired energy range
        """

        if energy_range is None:
            arf = self.effective_area
            energy_range = [arf.energy_thresh_lo, arf.energy_thresh_hi]

        energy_range = EnergyBounds(energy_range)
        ebounds = self.on_vector.energy_bounds
        if method == 'unbinned':
            on_list_temp = self.meta.on_list.select_energy(energy_range)
            off_list_temp = self.meta.off_list.select_energy(energy_range)
            on_vec = CountsSpectrum.from_eventlist(on_list_temp, ebounds)
            off_vec = CountsSpectrum.from_eventlist(off_list_temp, ebounds)
        elif method == 'binned':
            val = self.on_vector.energy_bounds.lower_bounds
            mask = np.invert(energy_range.contains(val))
            on_counts = np.copy(self.on_vector.counts)
            on_counts[mask] = 0
            off_counts = np.copy(self.off_vector.counts)
            off_counts[mask] = 0
            on_vec = CountsSpectrum(on_counts, ebounds)
            off_vec = CountsSpectrum(off_counts, ebounds)
        else:
            raise ValueError('Undefined method: {}'.format(method))

        off_vec.meta.update(backscal=self.off_vector.meta.backscal)
        m = copy.deepcopy(self.meta)
        m.update(energy_range=energy_range)

        return SpectrumObservation(self.obs_id, on_vec, off_vec,
                                   self.energy_dispersion, self.effective_area,
                                   meta=m)

    def _check_binning(self, **kwargs):
        """Check that ARF and RMF binnings are compatible
        """
        raise NotImplementedError


class SpectrumObservationList(list):
    """
    List of `~gammapy.spectrum.SpectrumObservation`.
    """

    def obs(self, obs_id):
        """Return one observation

        Parameters
        ----------
        obs_id : int
            Identifier
        """
        obs_id_list = [o.obs_id for o in self]
        idx = obs_id_list.index(obs_id)
        return self[idx]

    @property
    def total_spectrum(self):
        """Stack all observations belongig to the list"""
        return SpectrumObservation.stack_observation_list(self)

    def info(self):
        """Info string"""
        ss = " *** SpectrumObservationList ***"
        ss += "\n\nNumber of observations: {}".format(len(self))
        ss += "\nObservation IDs: {}".format([o.obs_id for o in self])

        return ss

    def filter_by_reflected_regions(self, n_min):
        """Filter observation list according to number of reflected regions.

        Condition: number of reflected regions >= nmin.

        Parameters
        ----------
        n_min : int
            Minimum number of reflected regions

        Returns
        -------
        idx : `~np.array`
            Indices of element fulfilling the condition
        """
        val = [o.off_vector.meta.backscal for o in self]

        condition = np.array(val) >= n_min
        idx = np.nonzero(condition)
        return idx[0]

    def write(self, outdir=None, **kwargs):
        """Create OGIP files

        Parameters
        ----------
        outdir : str, `~gammapy.extern.pathlib.Path`, optional
            Output directory, default: pwd
        """
        for obs in self:
            print (obs.off_vector.backscal)
            obs.write(outdir=outdir, **kwargs)

    # TODO: This should probably go away
    @classmethod
    def from_observation_table(cls, obs_table):
        """Create `~gammapy.spectrum.SpectrumObservationList` from an
        observation table.

        Parameters
        ----------
        obs_table : `~gammapy.data.ObservationTable`
            Observation table with column ``PHAFILE``
        """
        obs = [SpectrumObservation.read(_) for _ in obs_table['PHAFILE']]

        return cls(obs)


def stack(cls, specobs, group_id=None):
    """Stack `~gammapy.spectrum.SpectrumObservation`

    Observation stacking is implemented as follows
    Averaged exposure ratio between ON and OFF regions, arf and rmf
    :math:`\\alpha_{\\mathrm{tot}}`  for all observations is calculated as
    .. math:: \\alpha_{\\mathrm{tot}} = \\frac{\\sum_{i}\\alpha_i \\cdot N_i}{\\sum_{i} N_i}
    .. math:: \\arf_{\\mathrm{tot}} = \\frac{\\sum_{i}\\arf_i \\cdot \\livetime_i}{\\sum_{i} \\livetime_i}
    .. math:: \\rmf_{\\mathrm{tot}} = \\frac{\\sum_{i}\\rmf_i \\cdot arf_i \\cdot livetime_i}{\\sum_{i} arf_i \\cdot livetime_i}

    Parameters
    ----------
    obs_list : list of `~gammapy.spectrum.SpectrumObservations`
        Observations to stack
    group_id : int, optional
        ID for stacked observations

    Returns
    -------
    stacked_obs : `~gammapy.spectrum.SpectrumObservations`
    """

    group_id = obs_list[0].meta.obs_id if group_id is None else group_id

    # Stack ON and OFF vector using the _add__ method in the CountSpectrum class
    on_vec = np.sum([o.on_vector for o in obs_list])

    # If obs_list contains only on element np.sum does not call the
    #  _add__ method which lead to a faulty meta object
    if len(obs_list) == 1:
        on_vec.meta = Bunch(livetime=obs_list[0].meta.livetime,
                            backscal=1)

    on_vec.meta.update(obs_id=group_id)

    off_vec = np.sum([o.off_vector for o in obs_list])

    # Stack arf vector
    arf_band = [o.effective_area.data * o.meta.livetime.value for o in obs_list]
    arf_band_tot = np.sum(arf_band, axis=0)
    livetime_tot = np.sum([o.meta.livetime.value for o in obs_list])
    arf_vec = arf_band_tot / livetime_tot
    energy = obs_list[1].effective_area.energy.data
    data = arf_vec * obs_list[0].effective_area.data.unit

    arf = EffectiveAreaTable(energy=energy, data=data)

    # Stack rmf vector
    rmf_band = [o.energy_dispersion.pdf_matrix.T * o.effective_area.data.value * o.meta.livetime.value for
                o in obs_list]
    rmf_band_tot = np.sum(rmf_band, axis=0)
    pdf_mat = rmf_band_tot / arf_band_tot
    etrue = obs_list[0].energy_dispersion.true_energy
    ereco = obs_list[0].energy_dispersion.reco_energy
    inan = np.isnan(pdf_mat)
    pdf_mat[inan] = 0
    rmf = EnergyDispersion(pdf_mat.T, etrue, ereco)

    # Calculate average alpha
    alpha_band = [o.alpha * o.off_vector.total_counts for o in obs_list]
    alpha_band_tot = np.sum(alpha_band)
    off_tot = np.sum([o.off_vector.total_counts for o in obs_list])
    alpha_mean = alpha_band_tot / off_tot
    off_vec.meta.backscal = 1. / alpha_mean

    # Calculate energy range
    # TODO: for the moment we take the minimal safe energy range
    # Taking the whole range requires an energy dependent lifetime
    emin = max([_.meta.safe_energy_range[0] for _ in obs_list])
    emax = min([_.meta.safe_energy_range[1] for _ in obs_list])

    m = Bunch()
    m['energy_range'] = EnergyBounds([emin, emax])
    on_vec.meta['safe_energy_range'] = EnergyBounds([emin, emax])
    # m['safe_energy_range'] = EnergyBounds([emin, emax])
    m['obs_ids'] = [o.meta.obs_id for o in obs_list]
    m['alpha_method1'] = alpha_mean
    m['livetime'] = Quantity(livetime_tot, "s")
    m['group_id'] = group_id

    return cls(on_vec, off_vec, rmf, arf, meta=m)

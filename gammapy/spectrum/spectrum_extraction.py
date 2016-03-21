# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import os

import numpy as np
from astropy.coordinates import Angle
from astropy.table import Column, vstack
from astropy.units import Quantity

from . import CountsSpectrum
from .results import SpectrumStats
from ..background import ring_area_factor, Cube
from ..data import DataStore, ObservationTable
from ..extern.bunch import Bunch
from ..extern.pathlib import Path
from ..image import ExclusionMask
from ..irf import EffectiveAreaTable, EnergyDispersion
from ..region import SkyCircleRegion, find_reflected_regions
from ..utils.energy import EnergyBounds, Energy
from ..utils.scripts import make_path, write_yaml


__all__ = [
    'SpectrumExtraction',
    'SpectrumObservation',
    'SpectrumObservationList',
]

log = logging.getLogger(__name__)


class SpectrumExtraction(object):
    """Class for creating input data to 1D spectrum fitting

    The purpose of this class is to create 1D counts on and off counts vectors
    as well as an effective area vector and an energy dispersion matrix starting
    from an event list and 2D irfs for as set of observations. The container
    class for one specific observation is `~gammapy.spectrum.SpectrumObservation`.
    The present class is responsible for filling a list of such observations,
    starting from some extraction parameters.
    For more info see :ref:`spectral_fitting`.

    Parameters
    ----------
    datastore : `~gammapy.data.DataStore`
        Data for the analysis
    obs_table : `~gammapy.data.ObservationTable`
        Table of observations to analyse
    on_region : `gammapy.region.SkyCircleRegion`
        Circular region to extract on counts
    exclusion : `~gammapy.image.ExclusionMask`
        Exclusion regions
    bkg_method : dict
        Background method including necessary parameters
    nobs : int, optional
        number of observations to process
    ebounds : `~gammapy.utils.energy.EnergyBounds`, optional
        Reconstructed energy binning definition
    """

    OBSTABLE_FILE = 'observation_table.fits'
    EXCLUDEDREGIONS_FILE = 'excluded_regions.fits'
    REGIONS_FILE = 'regions.txt'
    TOTAL_STATS_FILE = 'total_spectrum_stats.yaml'
    ONLIST_FILE = 'on_list.fits'
    OFFLIST_FILE = 'off_list.fits'

    def __init__(self, datastore, obs_table, on_region, exclusion, bkg_method,
                 nobs=None, ebounds=None, **kwargs):

        self.on_region = on_region
        self.store = datastore
        self.exclusion = exclusion
        if ebounds is None:
            ebounds = EnergyBounds.equal_log_spacing(0.01, 316, 108, 'TeV')
        self.ebounds = ebounds
        self.bkg_method = bkg_method
        self.extra_info = kwargs
        self.obs_table = obs_table[0:nobs]

        self._observations = None

    def run(self, outdir=None):
        """Run all steps

        Extract spectrum, update observation table, filter observations, 
        write results to disk.

        Parameters
        ----------
        outdir : Path, str
            directory to write results files to
        """
        cwd = Path.cwd()
        outdir = cwd if outdir is None else make_path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        os.chdir(str(outdir))

        self.cutout_exclusion_mask()
        self.extract_spectrum()
        if self.bkg_method['type'] == 'reflected':
            self.filter_observations()

        self.observations.write_ogip()
        tot_stats = self.observations.total_spectrum.spectrum_stats
        tot_stats.to_yaml(self.TOTAL_STATS_FILE)
        self.write_configfile()
        self.write_regions()
        self.write_total_onlist()
        self.write_total_offlist()
        os.chdir(str(cwd))

    def filter_observations(self):
        """Filter observations by number of reflected regions"""
        n_min = self.bkg_method['n_min']
        obs = self.observations
        mask = obs.filter_by_reflected_regions(n_min)
        inv_mask = np.where([_ not in mask for _ in np.arange(len(mask + 1))])
        excl_obs = self.obs_table[inv_mask[0]]['OBS_ID'].data
        log.info('Excluding obs {} : Found less than {} reflected '
                 'region(s)'.format(excl_obs, n_min))
        self._observations = SpectrumObservationList(np.asarray(obs)[mask])
        self.obs_table = self.obs_table[mask]

    def extract_spectrum(self):
        """Extract 1D spectral information

        The result can be obtained via
        :func:`~gammapy.spectrum.spectrum_extraction.observations`
        The observation table is updated with some meta information.
        """

        observations = []

        t = self.obs_table
        for row in t:
            log.info('Extracting spectrum for observation {}'.format(row['OBS_ID']))
            temp = SpectrumObservation.from_datastore(row, self.store,
                                                      self.on_region,
                                                      self.bkg_method,
                                                      self.ebounds,
                                                      self.exclusion,
                                                      **self.extra_info
                                                      )

            observations.append(temp)

        self._observations = SpectrumObservationList(observations)

        #update meta info
        if 'OFFSET' not in t.colnames:
            offset = [_.meta.offset for _ in self.observations]
            t.add_column(Column(name='OFFSET', data=Angle(offset)))
        if 'CONTAINMENT' not in t.colnames:
            containment = [_.meta.containment for _ in self.observations]
            t.add_column(Column(name='CONTAINMENT', data=containment))
        if 'PHAFILE' not in t.colnames:
            pha = [str(_.meta.ogip_dir/_.meta.phafile) for _ in self.observations]
            t.add_column(Column(name='PHAFILE', data=pha))

    @property
    def observations(self):
        """`~gamampy.spectrum.ObservationList` of all observations

        This list is generated via
        :func:`~gammapy.spectrum.spectrum_extraction.extract_spectrum`
        when the property is first called and the result is cached.
        """
        if self._observations is None:
            self.extract_spectrum()
        return self._observations

    def copy(self, bkg_method=None):
        """Return copy of `~gammapy.spectrum.SpectrumExtraction`

        Parameters
        ----------
        bkg_method : dict, optional
            New background estimation method
        """

        bkg_method = self.bkg_method if bkg_method is None else bkg_method

        ana = SpectrumExtraction(datastore=self.store, obs_table=self.obs_table,
                                 on_region=self.on_region,
                                 bkg_method=bkg_method,
                                 exclusion=self.exclusion,
                                 ebounds=self.ebounds)
        return ana

    @classmethod
    def from_config(cls, config, **kwargs):
        """Create `~gammapy.spectrum.SpectrumAnalysis` from config dict

        Parameters
        ----------
        configfile : dict
            config dict
        """
        config = config['extraction']

        # Observations
        obs_table = ObservationTable.read(config['data']['obstable'])
        storename = config['data']['datastore']
        store = DataStore.from_all(storename)
        nobs = config['data']['nruns']
        nobs = None if nobs == 0 else nobs

        # Binning
        sec = config['binning']
        if sec['equal_log_spacing']:
            emin = Energy(sec['emin'])
            emax = Energy(sec['emax'])
            nbins = sec['nbins']
            ebounds = EnergyBounds.equal_log_spacing(
                emin, emax, nbins)
        else:
            vals = np.fromstring(sec['bounds'], dtype=float, sep=' ')
            ebounds = EnergyBounds(vals, sec['unit'])

        # ON region
        on_region = SkyCircleRegion.from_dict(config['on_region'])

        # OFF region
        bkg_method = config['off_region']

        # Exclusion
        excl_file = config['excluded_regions']['file']
        exclusion = ExclusionMask.read(excl_file)

        return cls(datastore=store, obs_table=obs_table, on_region=on_region,
                   bkg_method=bkg_method, exclusion=exclusion,
                   nobs=nobs, ebounds=ebounds, **kwargs)

    @classmethod
    def from_configfile(cls, configfile):
        """Create `~gammapy.spectrum.SpectrumExtraction` from configfile

        Parameters
        ----------
        configfile : str
            YAML config file
        """
        import yaml
        with open(configfile) as fh:
            config = yaml.safe_load(fh)

        return cls.from_config(config)

    def write_configfile(self, filename='config.yaml'):
        """Write config file in YAML format

        This is usefull when the `~gammapy.spectrum.SpectrumExtraction` has
        been altered and the changes want to be saved. Files need to read the
        configfile again are also written.

        Parameters
        ----------
        filename : str
            YAML file to write
        """
        config = dict(extraction=dict())
        e = config['extraction']
        e.update(data=dict())
        e['data']['obstable'] = self.OBSTABLE_FILE
        e['data']['datastore'] = str(self.store.hdu_table.base_dir)
        e['data']['nruns'] = len(self.obs_table)

        e.update(binning=dict())
        e['binning']['equal_log_spacing'] = False
        e['binning']['bounds'] = np.array_str(self.ebounds.value)[1:-1]
        e['binning']['unit'] = str(self.ebounds.unit)

        e.update(on_region=self.on_region.to_dict())
        e.update(off_region=self.bkg_method)
        e.update(excluded_regions=dict(file=self.EXCLUDEDREGIONS_FILE))

        log.info('Writing {}'.format(filename))
        write_yaml(config, filename=filename)
        log.info('Writing {}'.format(self.OBSTABLE_FILE))
        self.exclusion.write(self.EXCLUDEDREGIONS_FILE, clobber=True)
        self.obs_table.write(self.OBSTABLE_FILE, format='fits', overwrite=True)

    def write_regions(self):
        """Write ON and OFF regions to disk"""
        out = dict()
        out['on_region'] = self.on_region.to_dict()
        if self.bkg_method['type'] == 'reflected':
            out['off_region'] = dict()
            val = out['off_region']
            for obs in self.observations:
                val['obs_{}'.format(obs.meta.obs_id)] = obs.meta.off_region.to_dict()

        log.info('Writing {}'.format(self.REGIONS_FILE))
        write_yaml(out, self.REGIONS_FILE)

    def write_total_onlist(self):
        """Write event list containing ON events from all observations"""
        on_lists = [o.meta.on_list for o in self.observations]

        # Remove column TELMASK since raises an error when stacking it
        # see https://github.com/gammasky/hess-host-analyses/issues/34
        
        for l in on_lists:
            if(l.colnames=='TELMASK'):
                l.remove_column('TELMASK')

        total_list = vstack(on_lists, join_type='inner', metadata_conflicts='silent')
        total_list.meta = None
        log.info('Writing {}'.format(self.ONLIST_FILE))
        total_list.write(self.ONLIST_FILE, overwrite=True)

    def write_total_offlist(self):
        """Write event list containing OFF events from all observations"""
        off_lists = [o.meta.off_list for o in self.observations]

        # Remove column TELMASK since raises an error when stacking it
        # see https://github.com/gammasky/hess-host-analyses/issues/34
        for l in off_lists:
            if(l.colnames=='TELMASK'):
                l.remove_column('TELMASK')

        total_list = vstack(off_lists, join_type='exact', metadata_conflicts='silent')
        total_list.meta = None
        log.info('Writing {}'.format(self.OFFLIST_FILE))
        total_list.write(self.OFFLIST_FILE, overwrite=True)

    def cutout_exclusion_mask(self, fov='9 deg'):
        """Cutout appropriate part of exclusion mask

        In many cases the exclusion mask is given as all-sky image, but only a
        small fraction of that image is needed

        Parameters
        ----------
        exclusion : `~gammapy.image.ExclusionMask`
            Input exclusion mask
        fov : `~astropy.coordinates.Angle`
            Field of view
        """
        from astropy.nddata import Cutout2D
        from astropy.nddata.utils import PartialOverlapError
        fov = Angle(fov)
        exclusion = self.exclusion
        try:
            c = Cutout2D(exclusion.mask, self.on_region.pos, fov,
                         exclusion.wcs, copy=True, mode='strict')
        except PartialOverlapError:
            raise PartialOverlapError('FOV ({}) not completely contained '
                                      'in exclusion mask'.format(fov))

        self.exclusion = ExclusionMask(name=exclusion.name, data=c.data,
                                       wcs=c.wcs)

    def info(self):
        """Print some information
        """
        ss = "\nSpectrum Analysis\n"
        ss += "Observations : {}\n".format(len(self.observations))
        ss += "ON region    : {}\n".format(self.on_region.pos)

        return ss


class SpectrumObservation(object):
    """Storage class holding ingredients for 1D region based spectral analysis
    """

    def __init__(self, on_vector, off_vector, energy_dispersion,
                 effective_area, meta=None):
        self.on_vector = on_vector
        self.off_vector = off_vector
        self.energy_dispersion = energy_dispersion
        self.effective_area = effective_area
        self.meta = meta

    @classmethod
    def read_ogip(cls, phafile):
        """Read `~gammapy.spectrum.SpectrumObservation` from OGIP files.

        BKG file, ARF, and RMF must be set in the PHA header.

        Parameters
        ----------
        phafile : str
            OGIP PHA file to read
        """
        # Put here due to circular imports issues
        from ..irf import EnergyDispersion, EffectiveAreaTable

        f = make_path(phafile)
        base = f.parent
        on_vector = CountsSpectrum.read_pha(f)

        meta = on_vector.meta
        energy_dispersion = EnergyDispersion.read(str(base / meta.rmf))
        effective_area = EffectiveAreaTable.read(str(base / meta.arf))
        off_vector = CountsSpectrum.read_bkg(str(base / meta.bkg),
                                             str(base / meta.rmf))

        meta.update(phafile=phafile)
        return cls(on_vector, off_vector, energy_dispersion, effective_area,
                   meta)

    def write_ogip(self, phafile=None, bkgfile=None, rmffile=None, arffile=None,
                   outdir=None, clobber=True):
        """Write OGIP files

        The arf, rmf and bkg files are set in the :ref:`gadf:ogip-pha` FITS
        header. If no filenames are given, default names will be chosen.
       
        Parameters
        ----------
        phafile : `~gammapy.extern.pathlib.Path`, str
            PHA filename
        bkgfile : str
            BKG filename
        arffile : str
            ARF filename
        rmffile : str
            RMF : filename
        outdir : None
            directory to write the files to, default: pwd
        clobber : bool
            Overwrite
        """

        cwd = Path.cwd()
        outdir = cwd / make_path(self.meta.ogip_dir) if outdir is None else outdir
        outdir.mkdir(exist_ok=True, parents=True)

        if phafile is None:
            phafile = self.meta.phafile
        if arffile is None:
            arffile = phafile.replace('pha', 'arf')
        if rmffile is None:
            rmffile = phafile.replace('pha', 'rmf')
        if bkgfile is None:
            bkgfile = phafile.replace('pha', 'bkg')

        self.meta['phafile'] = str(outdir / phafile)

        self.on_vector.write(str(outdir / phafile), bkg=str(bkgfile), arf=str(arffile),
                             rmf=str(rmffile), clobber=clobber)
        self.off_vector.write(str(outdir / bkgfile), clobber=clobber)
        self.effective_area.write(str(outdir / arffile), energy_unit='keV',
                                  effarea_unit='cm2', clobber=clobber)
        self.energy_dispersion.write(str(outdir / rmffile), energy_unit='keV',
                                     clobber=clobber)


    @classmethod
    def from_datastore(cls, obs, store, on_region, bkg_method, ebounds,
                       exclusion, dry_run=False, calc_containment=False,
                       event_list=None):
        """ Create Spectrum Observation from datastore

        Meta info is stored in the input astropy table row

        Parameters
        ----------
        obs : `~astropy.table.Row`
            Row of a `~gammapy.spectrum.ObservationTable`
        store : `~gammapy.data.DataStore`
            Data Store
        on_region : `gammapy.region.SkyCircleRegion`
            Circular region to extract on counts
        bkg_method : dict
            Background method including necessary parameters
        ebounds : `~gammapy.utils.energy.EnergyBounds`
            Reconstructed energy binning definition
        exclusion : `~gammapy.image.ExclusionMask`
            Exclusion mask
        dry_run : bool, optional
            Only process meta data, not actual spectra are extracted
        calc_containment : bool, optional
            Calculate containment fraction of the on region
        event_list : `~gammapy.data.EventList`, optional
            Use event list different from the one in the data store
        """

        obs_id = obs['OBS_ID']

        if event_list is None:
            event_list = store.obs(obs_id=obs_id).events

        on = None
        off = None
        aeff = None
        edisp = None

        pointing = event_list.pointing_radec
        offset = pointing.separation(on_region.pos)
        livetime = store.obs(obs_id=obs_id).observation_live_time_duration

        if calc_containment:
            psf2d = store.obs(obs_id=obs_id).psf
            val = Energy('10 TeV')
            # TODO: error: unresolved reference: `m` on the next lines
            psf = psf2d.psf_at_energy_and_theta(val, m.offset)
            cont = psf.containment_fraction(m.on_region.radius)

        m = Bunch()
        m['obs_id'] = obs_id
        m['phafile'] = 'pha_run{}.fits'.format(obs_id)
        m['ogip_dir'] = Path.cwd() / 'ogip_data'
        m['offset'] = offset
        m['containment'] = 1
        m['livetime'] = livetime

        if dry_run:
            return cls(obs_id, None, None, None, None, meta=m)

        bkg_meta = Bunch()
        bkg_meta['bkg_method'] = bkg_method
        bkg_meta['on_region'] = on_region
        bkg_meta['exclusion'] = exclusion
        bkg_meta['pointing'] = pointing
        bkg_meta['ebounds'] = ebounds

        b = BackgroundEstimator(event_list, bkg_meta)
        b.make_off_vector()
        m['off_list'] = b.off_list
        m['off_region'] = b.off_region
        off_vec = b.off_vec
        off_vec.meta.update(backscal=b.backscal)
        off_vec.meta.update(livetime=livetime)

        m['on_list'] = event_list.select_circular_region(on_region)
        on_vec = CountsSpectrum.from_eventlist(m.on_list, ebounds)

        aeff2d = store.obs(obs_id=obs_id).aeff
        arf_vec = aeff2d.to_effective_area_table(offset)
        elo, ehi = arf_vec.energy_thresh_lo, arf_vec.energy_thresh_hi
        m['safe_energy_range'] = EnergyBounds([elo, ehi])

        edisp2d = store.obs(obs_id=obs_id).edisp
        rmf_mat = edisp2d.to_energy_dispersion(offset, e_reco=ebounds)

        # Todo: Define what metadata is stored where (obs table?)
        on_vec.meta.update(backscal=1)
        on_vec.meta.update(livetime=livetime)
        on_vec.meta.update(obs_id=obs_id)
        on_vec.meta.update(safe_energy_range=m.safe_energy_range)

        return cls(on_vec, off_vec, rmf_mat, arf_vec, meta=m)

    @classmethod
    def stack_observation_list(cls, obs_list, group_id=None):
        """Create `~gammapy.spectrum.SpectrumObservations` from list

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
        arf_band = [o.effective_area.effective_area * o.meta.livetime.value for o in obs_list]
        arf_band_tot = np.sum(arf_band, axis=0)
        livetime_tot = np.sum([o.meta.livetime.value for o in obs_list])
        arf_vec = arf_band_tot / livetime_tot
        ebounds = obs_list[0].effective_area.ebounds
        arf = EffectiveAreaTable(ebounds, Quantity(arf_vec, obs_list[0].effective_area.effective_area.unit))

        # Stack rmf vector
        rmf_band = [o.energy_dispersion.pdf_matrix.T * o.effective_area.effective_area.value * o.meta.livetime.value for o in obs_list]
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
        m['obs_ids'] = [o.meta.obs_id for o in obs_list]
        m['alpha_method1'] = alpha_mean
        m['livetime'] = Quantity(livetime_tot, "s")
        m['group_id'] = group_id

        return cls(on_vec, off_vec, rmf, arf, meta=m)

    @property
    def alpha(self):
        """Exposure ratio between ON and OFF region"""
        return self.on_vector.meta.backscal / self.off_vector.meta.backscal

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

    def restrict_energy_range(self, energy_range=None, method='binned'):
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

    def get_obslist_from_ids(self, id_list):
        """Return an subset of the observation list

        Parameters
        ----------
        id_list : list of int
            List of Observation Id (runnumber)

        Returns
        -------
        observation : `~gammapy.spectrum.SpectrumObservationList`
            Subset of observations
        """
        new_list = list()

        for id in id_list:
            ids = [o.meta.obs_id for o in self]
            try:
                i = ids.index(id)
            except ValueError:
                raise ValueError("Observation {} not in list".format(id))

            new_list.append(self[i])

        return SpectrumObservationList(new_list)

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

    def write_ogip(self, outdir=None, **kwargs):
        """Create OGIP files

        Parameters
        ----------
        outdir : str, `~gammapy.extern.pathlib.Path`
            write directory
        """
        for obs in self:
            obs.write_ogip(outdir=outdir, **kwargs)

    @classmethod
    def from_observation_table(cls, obs_table):
        """Create `~gammapy.spectrum.SpectrumObservationList` from an
        observation table.

        Parameters
        ----------
        obs_table : `~gammapy.data.ObservationTable`
            Observation table with column ``PHAFILE``
        """
        obs = [SpectrumObservation.read_ogip(_) for _ in obs_table['PHAFILE']]
        return cls(obs)


class BackgroundEstimator(object):
    """TBD

    Select events inside off regsion. At one point this can be replaced by a
    more generic `~gammapy.regions` module
    For available methods see :ref:`spectrum_background_method`

    Parameters
    ----------
    event_list : `~gammapy.data.EventList`
        Event list
    params : dict
        Necessary parameters
    """

    def __init__(self, event_list, params):
        self.event_list = event_list
        self.params = params
        m = self.params['bkg_method']['type']
        if m not in ['ring', 'reflected', 'bgmodel']:
            raise ValueError("Undefined background method: {}".format(m))

        self.off_list = None
        self.off_vec = None
        self.backscal = None
        self.off_region = None

    def make_off_vector(self):
        m = self.params['bkg_method']['type']
        if m == "ring":
            self._make_off_vector_ring()
        elif m == "reflected":
            self._make_off_vector_reflected()
        elif m == "bgmodel":
            self._make_off_vector_bgmodel()

    def _make_off_vector_reflected(self):
        """Helper function to create OFF vector from reflected regions"""
        kwargs = self.params['bkg_method'].copy()
        kwargs.pop('type')
        kwargs.pop('n_min')
        off = find_reflected_regions(self.params['on_region'],
                                     self.params['pointing'],
                                     self.params['exclusion'], **kwargs)
        off_list = self.event_list.select_circular_region(off)
        self.off_region = off
        self.backscal = len(off)
        self.off_list = off_list
        self.off_vec = CountsSpectrum.from_eventlist(off_list, self.params['ebounds'])

    def _make_off_vector_ring(self):
        """Helper function to create OFF vector from ring"""
        center = self.params['on_region'].pos
        radius = self.params['on_region'].radius
        m = self.params['bkg_method']
        inner = Angle(m['inner_radius'])
        outer = Angle(m['outer_radius'])
        off_list = self.event_list.select_sky_ring(center, inner, outer)
        self.backscal = ring_area_factor(radius.deg, inner.deg, outer.deg)
        self.off_list = off_list
        self.off_vec = CountsSpectrum.from_eventlist(off_list, self.params['ebounds'])

    def _make_off_vector_bgmodel(self, method):
        """Helper function to create OFF vector from BgModel"""
        s = self.params['datastore']
        filename = s.filename(obs_id=self.params.obs_id, filetype='background')
        cube = Cube.read(filename, scheme='bg_cube')
        # TODO: Properly transform to SkyCoords
        coords = Angle([self.params['offset'], '0 deg'])
        spec = cube.make_spectrum(coords, self.params['ebounds'])
        cnts = spec * self.params['ebounds'].bands * self.params['livetime'] * \
               self['params'].on_region.area
        off_vec = CountsSpectrum(cnts.decompose(), self.ebounds, backscal=1)
        self.backscal = 1
        self.off_vec = off_vec




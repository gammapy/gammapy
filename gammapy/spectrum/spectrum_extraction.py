# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import logging
import numpy as np
from astropy.units import Quantity
from astropy.coordinates import Angle, SkyCoord
from astropy.extern import six
from astropy.wcs.utils import skycoord_to_pixel
from . import CountsSpectrum
from .results import SpectrumStats
from ..extern.pathlib import Path
from ..extern.bunch import Bunch
from ..background import ring_area_factor, Cube
from ..data import DataStore, ObservationTable
from ..image import ExclusionMask
from ..region import SkyCircleRegion, find_reflected_regions
from ..utils.energy import EnergyBounds, Energy
from ..irf import EffectiveAreaTable, EnergyDispersion
from ..utils.scripts import (
    get_parser, set_up_logging_from_args, read_yaml, make_path,
)

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
    obs_ids : list, str
        List of observations or file containing such a list
    on_region : `gammapy.region.SkyCircleRegion`
        Circular region to extract on counts


    exclusion : `~gammapy.image.ExclusionMask`
        Exclusion regions
    bkg_method : dict
        Background method including necessary parameters
    nobs : int
        number of observations to process, 0 means all observations
    ebounds : `~gammapy.utils.energy.EnergyBounds`, optional
        Reconstructed energy binning definition
    """

    def __init__(self, datastore, obs_ids, on_region, exclusion, bkg_method,
                 nobs=-1, ebounds=None, **kwargs):

        self.on_region = on_region
        self.store = datastore
        self.exclusion = exclusion
        if ebounds is None:
            ebounds = EnergyBounds.equal_log_spacing(0.1, 10, 20, 'TeV')
        self.ebounds = ebounds
        self.bkg_method = bkg_method
        self.nobs = nobs
        self.extra_info = kwargs

        if isinstance(obs_ids, six.string_types):
            temp = make_path(obs_ids)
            obs_ids = np.loadtxt(str(temp), dtype=np.int)
        self.obs_ids = obs_ids

        self._observations = None

    def run(self):
        """Run all steps

        Extract spectrum, filter observations, write results to disk.
        """
        self.extract_spectrum()
        if self.bkg_method['type'] == 'reflected':
            self.filter_observations()

        o = self.observations
        o.write_ogip_data('ogip_data')
        o.total_spectrum.spectrum_stats.to_yaml('total_spectrum_stats.yaml')
        o.to_observation_table().write('observation_table.fits', format='fits',
                                       overwrite=True)

    def filter_observations(self):
        """Filter observations by number of reflected regions"""
        obs = self.observations
        mask = obs.filter_by_reflected_regions(self.bkg_method['n_min'])
        self._observations = SpectrumObservationList(np.asarray(obs)[mask])

    def extract_spectrum(self, nobs=None):
        """Extract 1D spectral information

        The result can be obtained via
        :func:`~gammapy.spectrum.spectrum_extraction.observations`
        """
        nobs = self.nobs if nobs is None else nobs
        observations = []
        for i, val in enumerate(np.atleast_1d(self.obs_ids)):
            log.info('Extracting spectrum for observation {}'.format(val))
            try:
                temp = SpectrumObservation.from_datastore(val, self.store,
                                                          self.on_region,
                                                          self.bkg_method,
                                                          self.ebounds,
                                                          self.exclusion,
                                                          **self.extra_info
                                                          )
            except IndexError as err:
                log.warning(
                    'Could not load observation {} from store{}'
                    'Error: \n{}'.format(val, self.store.base_dir, err))
                nobs += 1
                continue
                
            observations.append(temp)
            if i == nobs - 1:
                break

        self._observations = SpectrumObservationList(observations)

        if len(self.observations) == 0:
            raise ValueError("No valid observations found")

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

        ana = SpectrumExtraction(datastore=self.store, obs_ids=self.obs_ids,
                                 on_region=self.on_region,
                                 bkg_method=bkg_method,
                                 exclusion=self.exclusion, nobs=0,
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
        obs = config['data']['runlist']
        storename = config['data']['datastore']
        store = DataStore.from_all(storename)
        nobs = config['data']['nruns']

        # Binning
        sec = config['binning']
        if sec['equal_log_spacing']:
            emin = Energy(sec['emin'])
            emax = Energy(sec['emax'])
            nbins = sec['nbins']
            ebounds = EnergyBounds.equal_log_spacing(
                emin, emax, nbins)
        else:
            if sec['binning'] is None:
                raise ValueError("No binning specified")

        # ON region
        radius = Angle(config['on_region']['radius'])
        x = config['on_region']['center_x']
        y = config['on_region']['center_y']
        frame = config['on_region']['system']
        center = SkyCoord(x, y, frame=frame)
        on_region = SkyCircleRegion(center, radius)

        # OFF region
        bkg_method = config['off_region']

        # Exclusion
        excl_file = config['excluded_regions']['file']
        exclusion = ExclusionMask.from_fits(excl_file)

        return cls(datastore=store, obs_ids=obs, on_region=on_region,
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

    def __init__(self, obs_id, on_vector, off_vector, energy_dispersion,
                 effective_area, meta=None):
        self.obs_id = obs_id
        self.on_vector = on_vector
        self.off_vector = off_vector
        self.energy_dispersion = energy_dispersion
        self.effective_area = effective_area
        self.meta = Bunch(meta) if meta is not None else Bunch()

        # These values are needed for I/O
        self.meta.setdefault('phafile', 'None')

    @classmethod
    def read_ogip(cls, phafile):
        """ Read `~gammapy.spectrum.SpectrumObservation` from OGIP files

        BKG file, ARF, and RMF must be set in the PHA header

        Parameters
        ----------
        phafile : str
            OGIP PHA file to read
        """
        # Put here due to circular imports issues
        from ..irf import EnergyDispersion, EffectiveAreaTable

        f = make_path(phafile)
        base = f.parent
        on_vector = CountsSpectrum.read(f)

        meta = on_vector.meta
        energy_dispersion = EnergyDispersion.read(str(base / meta.RESPFILE))
        effective_area = EffectiveAreaTable.read(str(base / meta.ANCRFILE))
        off_vector = CountsSpectrum.read(str(base / meta.BACKFILE),
                                         str(base / meta.RESPFILE))

        meta.update(phafile=phafile)
        return cls(meta.OBS_ID, on_vector, off_vector, energy_dispersion,
                   effective_area, meta)

    @classmethod
    def from_datastore(cls, obs_id, store, on_region, bkg_method, ebounds,
                       exclusion, save_meta=True, dry_run=False, calc_containment=False):
        """ Create Spectrum Observation from datastore

        Extraction parameters are stored in the meta attribute

        Parameters
        ----------
        obs : int
            Observation ID, runnumber
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
        save_meta : bool, optional
            Save meta information, default: True
        dry_run : bool, optional
            Only process meta data, not actual spectra are extracted
        calc_containment : bool, optional
            Calculate containment fraction of the on region
        """

        event_list = store.load(obs_id=obs_id, filetype='events')
        on = None
        off = None
        aeff = None
        edisp = None

        m = Bunch()
        m['pointing'] = event_list.pointing_radec
        m['offset'] = m.pointing.separation(on_region.pos)
        m['livetime'] = event_list.observation_live_time_duration
        m['exclusion'] = exclusion
        m['on_region'] = on_region
        m['bkg_method'] = bkg_method
        m['datastore'] = store
        m['ebounds'] = ebounds
        m['obs_id'] = obs_id
        m['zen'] = 90 - event_list.meta['ALT_PNT']
        m['coszen'] = np.cos(m['zen'] * np.pi / 180.)
        # m['muoneff'] = event_list.meta['MUONEFF']

        if calc_containment:
            psf2d = store.load(obs_id=obs_id, filetype='psf')
            val = Energy('10 TeV')
            psf = psf2d.psf_at_energy_and_theta(val, m.offset)
            cont = psf.containment_fraction(m.on_region.radius)
            m['psf_containment'] = float(cont)

        if dry_run:
            return cls(obs_id, None, None, None, None, meta=m)

        b = BackgroundEstimator(event_list, m)
        b.make_off_vector()
        m['off_list'] = b.off_list
        m['off_region'] = b.off_region
        off_vec = b.off_vec
        off_vec.meta.update(backscal=b.backscal)
        off_vec.meta.update(livetime=m.livetime)

        m['on_list'] = event_list.select_circular_region(on_region)
        on_vec = CountsSpectrum.from_eventlist(m.on_list, ebounds)

        aeff2d = store.load(obs_id=obs_id, filetype='aeff')
        arf_vec = aeff2d.to_effective_area_table(m.offset)
        elo, ehi = arf_vec.energy_thresh_lo, arf_vec.energy_thresh_hi
        m['safe_energy_range'] = EnergyBounds([elo, ehi])

        edisp2d = store.load(obs_id=obs_id, filetype='edisp')
        rmf_mat = edisp2d.to_energy_dispersion(m.offset, e_reco=ebounds)

        m = None if not save_meta else m

        # Todo: Agree where to store all meta info
        on_vec.meta.update(m)

        return cls(obs_id, on_vec, off_vec, rmf_mat, arf_vec, meta=m)

    @classmethod
    def grouping_from_an_observation_list(cls, obs_list, obs_stacked_id):

        """Create `~gammapy.spectrum.SpectrumObservations` from list

       Observation stacking is implemented as follows

       Averaged exposure ratio between ON and OFF regions

       :math:`\\alpha_{\\mathrm{tot}}` for all observations is calculated as

       .. math:: \\alpha_{\\mathrm{tot}} = \\frac{\\sum_{i}\\alpha_i \\cdot N_i}{\\sum_{i} N_i}

       where :math:`N_i` is the number of OFF counts for observation :math:`i`

       Parameters
       ----------
       obs_list : list of `~gammapy.spectrum.SpectrumObservations`
           Observations to stack
       obs_id : int, optional
           Observation ID for stacked observations
       """
        # Stack ON and OFF vector using the _add__ method in the CountSpectrum class
        on_vec = np.sum([o.on_vector for o in obs_list])
        off_vec = np.sum([o.off_vector for o in obs_list])
        rmf = None

        arf_band = [o.effective_area.effective_area * o.meta.livetime.value for o in obs_list]
        arf_band_tot = np.sum(arf_band, axis=0)
        livetime_tot = np.sum([o.meta.livetime.value for o in obs_list])
        arf_vec = arf_band_tot / livetime_tot
        ener_hi = obs_list[0].effective_area.energy_hi
        ener_lo = obs_list[0].effective_area.energy_lo
        arf = EffectiveAreaTable(ener_lo, ener_hi, Quantity(arf_vec, obs_list[0].effective_area.effective_area.unit))

        """
        # Stack rmf vector
        #Je crois que je dois mettre un .T si je multiplie rmf_mat avec arf car dim rmf_mat is (Etrue,Ereco) et pour multiplier un tableau 2D avec un 1D de dim Etrue, le tableau 2D doit avec la dim (Ereco, Etrue) mais a verifier ladimension de rmf_mat
        rmf_band = [o.rmf_mat.T *o.arf_vector * o.livetime for o in obs_list]
        #ATTENTION: DANS LE np.sum IL FAUT FAIRE SELON LA DIMENSION DES OBSERVATIONS CAR IL Y A LA DIMENSION DES ENERGIES true et des energies reco ICI. Donc rmf_band est a 3D (voir quelle est la dimension des observations)
        rmf_band_tot = np.sum(rmf_band, axis=dimension_observation)
        #ici o.arf_vector*o.livetime est a 2D (dim_E_true*dim_list_observation)
        livetime_arf_tot = np.sum([o.arf_vector*o.livetime for o in obs_list], axis=dimension_observation)
        #Dim de rmf_band_tot est 2D (Etrue,Ereco) ou (Ereco,Etrue). Voir dans quelle sens est la shape pour voir si je peux diviser par un truc de la shape Etrue ou si je dois mettre un .T a rmf_band_tot
        rmf_mat=rmf_band_tot/livetime_arf_tot
        """

        # Calculate average alpha
        alpha_band = [o.alpha * o.off_vector.total_counts for o in obs_list]
        alpha_band_tot = np.sum(alpha_band)
        off_tot = np.sum([o.off_vector.total_counts for o in obs_list])
        alpha_mean = alpha_band_tot / off_tot
        off_vec.meta.backscal = 1. / alpha_mean

        # Calculate energy range
        # TODO: pour l instant on va prendre le plus petit range en energy possible pour pas se faire chier avec des
        # livetime different en fonctiond des bins en erngies mais c'est crado. Voir avec Regis aussi c'est quoi cette
        #  energie range et si on a vraiment besoin de prendre le max en energy range et de definir un livetime
        # dependant des energies bins
        emin = max([_.meta.energy_range[0] for _ in obs_list])
        emax = min([_.meta.energy_range[1] for _ in obs_list])

        m = Bunch()
        m['energy_range'] = EnergyBounds([emin, emax])
        m['obs_ids'] = [o.obs_id for o in obs_list]
        m['alpha_method1'] = alpha_mean
        m['livetime'] = Quantity(livetime_tot, "s")
        #import IPython; IPython.embed()
        return cls(obs_stacked_id, on_vec, off_vec, rmf, arf,  meta=m)

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
        outdir = cwd if outdir is None else cwd / make_path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        if phafile is None:
            phafile = "pha_run{}.pha".format(self.obs_id)
        if arffile is None:
            arffile = "arf_run{}.fits".format(self.obs_id)
        if rmffile is None:
            rmffile = "rmf_run{}.fits".format(self.obs_id)
        if bkgfile is None:
            bkgfile = "bkg_run{}.fits".format(self.obs_id)

        self.meta['phafile'] = str(outdir / phafile)

        self.on_vector.write(str(outdir / phafile), bkg=str(bkgfile), arf=str(arffile),
                             rmf=str(rmffile), clobber=clobber)
        self.off_vector.write(str(outdir / bkgfile), clobber=clobber)
        self.effective_area.write(str(outdir / arffile), energy_unit='keV',
                                  effarea_unit='cm2', clobber=clobber)
        self.energy_dispersion.write(str(outdir / rmffile), energy_unit='keV',
                                     clobber=clobber)

    def plot_exclusion_mask(self, size=None, **kwargs):
        """Plot exclusion mask for this observation

        The plot will be centered at the pointing position

        Parameters
        ----------
        size : `~astropy.coordinates.Angle`
            Edge length of the plot
        """
        size = Angle('5 deg') if size is None else Angle(size)
        ax = self.meta.exclusion.plot(**kwargs)
        self._set_ax_limits(ax, size)
        point = skycoord_to_pixel(self.meta.pointing, ax.wcs)
        ax.scatter(point[0], point[1], s=250, marker="+", color='black')
        return ax

    def plot_on_region(self, ax=None, **kwargs):
        """Plot target regions"""
        ax = self.plot_exclusion_mask() if ax is None else ax
        self.meta.on_region.plot(ax, **kwargs)
        return ax

    def plot_reflected_regions(self, ax=None, **kwargs):
        """Plot reflected regions"""
        ax = self.plot_exclusion_mask() if ax is None else ax
        self.meta.off_region.plot(ax, **kwargs)
        return ax

    def _check_binning(self, **kwargs):
        """Check that ARF and RMF binnings are compatible
        """
        pass

    def _set_ax_limits(self, ax, extent):

        if 'GLAT' in ax.wcs.to_header()['CTYPE2']:
            center = self.meta.pointing.galactic
            xlim = (center.l + extent / 2).value, (center.l - extent / 2).value
            ylim = (center.b + extent / 2).value, (center.b - extent / 2).value
        else:
            center = self.meta.pointing.icrs
            xlim = (center.ra + extent / 2).value, (center.ra - extent / 2).value
            ylim = (center.dec + extent / 2).value, (center.dec - extent / 2).value

        limits = ax.wcs.wcs_world2pix(xlim, ylim, 1)
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])


class SpectrumObservationList(list):
    """List of `~gammapy.spectrum.SpectrumObservation`
    """

    # @classmethod
    def get_obslist_from_obsid(self, list_ids):
        """Return an observation with a certain id

        Parameters
        ----------
        list_id : list of int
            List of Observation Id (runnumber)

        Returns
        -------
        observation : `~gammapy.spectrum.SpectrumObservationList`
            List of `~gammapy.spectrum.SpectrumObservation`
        """
        new_list = list()

        for id in list_ids:
            ids = [o.obs_id for o in self]
            try:
                i = ids.index(id)
            except ValueError:
                raise ValueError("Observation {} not in list".format(id))

            new_list.append(self[i])

        return SpectrumObservationList(new_list)

    @property
    def total_spectrum(self):
        return SpectrumObservation.from_observation_list(self)

    def info(self):
        """Info string"""
        ss = " *** SpectrumObservationList ***"
        ss += "\n\nNumber of observations: {}".format(len(self))
        ss += "\nObservation IDs: {}".format([o.obs_id for o in self])

        return ss

    def filter_by_reflected_regions(self, n_min):
        """Filter observation list according to number of reflected regions

        Condition: number of reflected regions >= nmin

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

    def write_ogip_data(self, outdir, **kwargs):
        """Create OGIP files

        Parameters
        ----------
        outdir : str, `~gammapy.extern.pathlib.Path`
            write directory
        """
        for obs in self:
            obs.write_ogip(outdir=outdir, **kwargs)

    @classmethod
    def read_ogip(cls, dir='ogip_data'):
        """Read `~gammapy.spectrum.SpectrumObservationList` from OGIP files

        The pha file need to be contained in one directroy and have '.pha' as
        suffix

        Parameters
        ----------
        dir : str, Path
            Directory holding the OGIP data
        """
        dir = make_path(dir)
        obs = [SpectrumObservation.read_ogip(_) for _ in dir.glob('*.pha')]
        return cls(obs)

    def to_observation_table(self):
        """Create `~gammapy.data.ObservationTable`"""
        names = ['OBS_ID', 'PHAFILE', 'OFFSET']
        col1 = [o.obs_id for o in self]
        col2 = [o.meta.phafile for o in self]
        col3 = [o.meta.offset.value for o in self]
        return ObservationTable(data=[col1, col2, col3], names=names)


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


<<<<<<< HEAD
=======
def run_spectrum_extraction_using_config(config, **kwargs):
    """
    Run a 1D spectral analysis using a config dict

    kwargs are forwarded to
     :func:`spectrum.spectrum_extraction.SpectrumObservation.from_config`

    Parameters
    ----------
    config : dict
        Config dict

    Returns
    -------
    analysis : `~gammapy.spectrum.spectrum_extraction.SpectrumExtraction`
        Spectrum extraction analysis instance
    """
    kwargs.setdefault('dry_run', False)
    config = config['extraction']
    outdir = config['results']['outdir']
    log.info("\nStarting analysis {}".format(outdir))
    outdir = make_path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    analysis = SpectrumExtraction.from_config(config, **kwargs)
    obs = analysis.observations
    if kwargs['dry_run']:
        return analysis

    if config['off_region']['type'] == 'reflected':
        mask = obs.filter_by_reflected_regions(config['off_region']['n_min'])
        # Todo: should ObservationList subclass np.array to avoid this hack?
        temp = np.asarray(obs)[mask]
        obs = SpectrumObservationList(temp)

    obs_in_erange = SpectrumObservationList(
        [o.restrict_energy_range(method='binned') for o in obs])

    # Output
    if config['results']['write_ogip']:
        obs.write_ogip_data(str(outdir / 'ogip_data'))

    rfile = outdir / config['results']['result_file']
    obs.total_spectrum.spectrum_stats.to_yaml(str(rfile))
    obs_in_erange.total_spectrum.spectrum_stats.to_yaml('test.yaml')
    log.info('\nWriting file {}'.format(rfile))
    obs.to_observation_table().write(
        str(outdir / 'observations.fits'), format='fits', overwrite=True)

    return analysis
>>>>>>> arf seems ok

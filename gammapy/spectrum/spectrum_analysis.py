# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy.extern import six
from astropy.wcs.utils import skycoord_to_pixel

from . import CountsSpectrum
from .results import SpectrumStats
from ..extern.pathlib import Path
from ..extern.bunch import Bunch
from ..background import ring_area_factor, Cube
from ..data import DataStore
from ..image import ExclusionMask
from ..region import SkyCircleRegion, find_reflected_regions
from ..utils.energy import EnergyBounds, Energy
from ..utils.scripts import (
    get_parser, set_up_logging_from_args, read_yaml, make_path,
)

__all__ = [
    'SpectrumAnalysis',
    'SpectrumObservation',
    'SpectrumFit',
    'run_spectral_fit_using_config',
]

log = logging.getLogger(__name__)


def main(args=None):
    parser = get_parser(SpectrumAnalysis)
    parser.add_argument('config_file', type=str,
                        help='Config file in YAML format')
    parser.add_argument("-l", "--loglevel", default='info',
                        choices=['debug', 'info', 'warning', 'error',
                                 'critical'],
                        help="Set the logging level")

    args = parser.parse_args(args)
    set_up_logging_from_args(args)

    config = read_yaml(args.config_file)
    run_spectral_fit_using_config(config)


class SpectrumAnalysis(object):
    """Class for 1D spectrum fitting

    This class creates several `~gammapy.spectrum.SpectrumObservation` instances
    for a given list of observations. All functionality is implemented in the
    `~gammapy.spectrum.SpectrumObservation` class. This class is only
    responsible for managing several observations.
    For more info see :ref:`spectral_fitting`.

    Parameters
    ----------
    datastore : `~gammapy.data.DataStore`
        Data for the analysis
    obs : list, str
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

    def __init__(self, datastore, obs, on_region, exclusion, bkg_method=None,
                 nobs=-1, ebounds=None):

        self.on_region = on_region
        self.store = datastore
        self.exclusion = exclusion
        if ebounds is None:
            ebounds = EnergyBounds.equal_log_spacing(0.1, 10, 20, 'TeV')
        self.ebounds = ebounds
        self.bkg_method = 'no method' if bkg_method is None else bkg_method

        if isinstance(obs, six.string_types):
            obs = make_path(obs)
            obs = np.loadtxt(str(obs), dtype=np.int)

        observations = []
        for i, val in enumerate(np.atleast_1d(obs)):
            try:
                temp = SpectrumObservation.from_datastore(val, self.store,
                                                          on_region,
                                                          bkg_method, ebounds,
                                                          exclusion)
            except IndexError:
                log.warning(
                    'Observation {} not in store {}'.format(val, datastore))
                nobs += 1
                continue
            observations.append(temp)
            if i == nobs - 1:
                break

        self._observations = np.array(observations)

        if len(self.observations) == 0:
            raise ValueError("No valid observations found")
        if bkg_method['type'] == 'reflected':
            mask = self.filter_by_reflected_regions(bkg_method['n_min'])
            self._observations = self.observations[mask]

    def copy(self, bkg_method=None):
        """Return copy of `~gammapy.spectrum.SpectrumAnalysis`

        Parameters
        ----------
        bkg_method : dict, optional
            New background estimation method
        """

        bkg_method = self.bkg_method if bkg_method is None else bkg_method

        ana = SpectrumAnalysis(datastore=self.store, obs=self.obs_ids,
                               on_region=self.on_region, bkg_method=bkg_method,
                               exclusion=self.exclusion, nobs=0,
                               ebounds=self.ebounds)
        return ana

    @property
    def observations(self):
        """`np.array` of all observations belonging to the analysis
        """
        return self._observations

    @property
    def obs_ids(self):
        """List of all observation ids"""
        return [o.obs for o in self.observations]

    def get_obs_by_id(self, id):
        """Return an certain observation belonging to the analysis

        Parameters
        ----------
        id : int
            Observation Id (runnumber)

        Returns
        -------
        observation : `~gammapy.spectrum.SpectrumObservation`
            Spectrum observation
        """
        try:
            i = self.obs_ids.index(id)
        except ValueError:
            raise ValueError("Observation {} not found".format(id))
        return self.observations[i]

    # Todo: move to spectrum stats
    @property
    def total_alpha(self):
        """Averaged exposure ratio between ON and OFF regions

        :math:`\\alpha_{\\mathrm{tot}}` for all observations is calculated as

        .. math:: \\alpha_{\\mathrm{tot}} = \\frac{\\sum_{i}\\alpha_i \\cdot N_i}{\\sum_{i} N_i}

        where :math:`N_i` is the number of OFF counts for observation :math:`i`
        """
        val = [o.alpha * o.off_vector.total_counts for o in self.observations]
        num = np.sum(val)
        den = np.sum([o.off_vector.total_counts for o in self.observations])
        return num/den

    @property
    def total_spectrum(self):
        return SpectrumObservation.from_observation_list(self.observations)

    @classmethod
    def from_config(cls, config):
        """Create `~gammapy.spectrum.SpectrumAnalysis` from config dict

        Parameters
        ----------
        configfile : dict
            config dict
        """

        # Observations
        obs = config['general']['runlist']
        storename = config['general']['datastore']
        store = DataStore.from_all(storename)
        nobs = config['general']['nruns']

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

        # Outdir
        outdir = config['general']['outdir']

        return cls(datastore=store, obs=obs, on_region=on_region,
                   bkg_method=bkg_method, exclusion=exclusion,
                   nobs=nobs, ebounds=ebounds)

    @classmethod
    def from_configfile(cls, configfile):
        """Create `~gammapy.spectrum.SpectrumAnalysis` from configfile

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

    def write_ogip_data(self, outdir, **kwargs):
        """Create OGIP files

        Parameters
        ----------
        outdir : str, `~gammapy.extern.pathlib.Path`
            write directory
        """

        for obs in self.observations:
            obs.write_ogip(outdir=outdir, **kwargs)
            log.info('Creating OGIP data for run{}'.format(obs.obs_id))

    def filter_by_reflected_regions(self, n_min):
        """Filter runs according to number of reflected regions

        Condition: number of reflected regions >= nmin

        Parameters
        ----------
        n_min : int
            Minimum number of reflected regions

        Returns
        -------
        idx : `np.array`
            Indices of element fulfilling the condition
        """
        val = self.bkg_method['type']
        if val != 'reflected':
            raise ValueError("Wrong background method: {}".format(val))

        val = [o.off_vector.backscal for o in self.observations]
        condition = np.array(val) >= n_min
        idx = np.nonzero(condition)
        return idx[0]


class SpectrumObservation(object):
    """Storage class holding ingredients for 1D region based spectral analysis
    """

    def __init__(self, obs_id, on_vector, off_vector, energy_dispersion,
                 effective_area, meta):
        self.obs_id = obs_id
        self.on_vector = on_vector
        self.off_vector = off_vector
        self.energy_dispersion = energy_dispersion
        self.effective_area = effective_area
        self.meta = meta

    @classmethod
    def read_ogip(cls, phafile, rmffile=None, bkgfile=None, arffile=None):
        """ Read PHA file

        Parameters
        ----------
        phafile : str
            OGIP PHA file to read
        """
        pass

    @classmethod
    def from_datastore(cls, obs_id, store, on_region, bkg_method, ebounds,
        exclusion, save_meta=True):
        """ Create Spectrum Observation from datastore

        BLABLA is stored in the meta

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

        b = BackgroundEstimator(event_list, m)
        b.make_off_vector()
        m['off_list'] = b.off_list
        m['off_region'] = b.off_region
        off_vec = b.off_vec
        off_vec.backscal = b.backscal

        m['on_list'] = event_list.select_circular_region(on_region)
        on_vec = CountsSpectrum.from_eventlist(m.on_list, ebounds)

        aeff2d = store.load(obs_id=obs_id, filetype='aeff')
        arf_vec = aeff2d.to_effective_area_table(m.offset)

        edisp2d = store.load(obs_id=obs_id, filetype='edisp')
        rmf_mat = edisp2d.to_energy_dispersion(m.offset, e_reco=ebounds)

        m = None if not save_meta else m

        return cls(obs_id, on_vec, off_vec, rmf_mat, arf_vec, meta=m)

    @classmethod
    def from_observation_list(cls, obs_list, obs_id = None):
        """Create `~gammapy.spectrum.SpectrumObservations` from list

        Observation stacking is implemented as follows

        Parameters
        ----------
        obs_list : list of `~gammapy.spectrum.SpectrumObservations`
            Observations to stack
        obs_id : int, optional
            Observation ID for stacked observations
        """
        obs_id = 0 if obs_id is None else obs_id

        on_vec = np.sum([o.on_vector for o in obs_list])
        off_vec = np.sum([o.off_vector for o in obs_list])
        # Todo : Stack RMF and ARF
        arf = None
        rmf = None

        m = Bunch()
        m['obs_ids'] = [o.obs_id for o in obs_list]
        return cls(obs_id, on_vec, off_vec, arf, rmf, meta=m)

    @property
    def alpha(self):
        """Exposure ratio between ON and OFF region"""
        return self.on_vector.backscal / self.off_vector.backscal

    @property
    def spectrum_stats(self):
        return SpectrumStats.from_spectrum_observation(self)

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

        outdir = Path.cwd() if outdir is None else make_path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        if phafile is None:
            phafile = "pha_run{}.pha".format(self.obs_id)
        if arffile is None:
            arffile = "arf_run{}.fits".format(self.obs_id)
        if rmffile is None:
            rmffile = "rmf_run{}.fits".format(self.obs_id)
        if bkgfile is None:
            bkgfile = "bkg_run{}.fits".format(self.obs_id)

        self.on_vector.write(str(outdir/phafile), bkg=str(bkgfile), arf=str(arffile),
                             rmf=str(rmffile), clobber=clobber)
        self.off_vector.write(str(outdir/bkgfile), clobber=clobber)
        self.effective_area.write(str(outdir/arffile), energy_unit='keV',
                                  effarea_unit='cm2', clobber=clobber)
        self.energy_dispersion.write(str(outdir/rmffile), energy_unit='keV',
                                     clobber=clobber)

    def plot_exclusion_mask(self, size=None, **kwargs):
        """Plot exclusion mask for this observation

        The plot will be centered at the pointing position

        Parameters
        ----------
        size : `~astropy.coordinates.Angle`
            Edge length of the plot
        """
        size = Angle('5 deg') if size is None else size
        ax = self.exclusion.plot(**kwargs)
        self._set_ax_limits(ax, size)
        point = skycoord_to_pixel(self.pointing, ax.wcs)
        ax.scatter(point[0], point[1], s=250, marker="+", color='black')
        return ax

    def plot_on_region(self, ax=None, **kwargs):
        """Plot target regions"""
        ax = self.plot_exclusion_mask() if ax is None else ax
        self.on_region.plot(ax, **kwargs)

    def plot_reflected_regions(self, ax=None, **kwargs):
        """Plot reflected regions"""
        ax = self.plot_exclusion_mask() if ax is None else ax
        if self.reflected_regions is None:
            self.make_reflected_regions()
        self.reflected_regions.plot(ax, **kwargs)

    def _check_binning(self, **kwargs):
        """Check that ARF and RMF binnings are compatible
        """
        pass

    def _set_ax_limits(self, ax, extent):

        if 'GLAT' in ax.wcs.to_header()['CTYPE2']:
            center = self.pointing.galactic
            xlim = (center.l + extent/2).value, (center.l - extent/2).value
            ylim = (center.b + extent/2).value, (center.b - extent/2).value
        else:
            center = self.pointing.icrs
            xlim = (center.ra + extent/2).value, (center.ra - extent/2).value
            ylim = (center.dec + extent/2).value, (center.dec - extent/2).value

        limits = ax.wcs.wcs_world2pix(xlim, ylim,1)
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])


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


# Todo: put fitting functionality into separate file

class SpectrumFit(object):
    """
    Spectral Fit

    Parameters
    ----------
    pha : list of str, `~gammapy.extern.pathlib.Path`
        List of PHA files to fit
    """

    DEFAULT_STAT = 'wstat'

    def __init__(self, pha, bkg=None, arf=None, rmf=None, stat=DEFAULT_STAT):

        self.pha = [make_path(f) for f in pha]
        self._model = None
        self._thres_lo = None
        self._thres_hi = None
        self.statistic = stat

    @classmethod
    def from_config(cls, config):
        """Create `~gammapy.spectrum.SpectrumFit` from config file"""
        outdir = make_path(config['general']['outdir'])
        # TODO: this is not a good solution! an obs table should be used
        return cls.from_dir(outdir/'ogip_data')

    @classmethod
    def from_dir(cls, dir):
        """Create `~gammapy.spectrum.SpectrumFit` using directory

        All PHA files in the directory will be used
        """
        dir = make_path(dir)
        pha_list = dir.glob('pha_run*.pha')
        return cls(pha_list)

    @property
    def model(self):
        """
        Spectral model to be fit
        """
        return self._model

    @model.setter
    def model(self, model, name=None):
        """

        Parameters
        ----------
        model : `~sherpa.models.ArithmeticModel`
            Fit model
        name : str
            Name for Sherpa model instance, optional
        """
        import sherpa.models

        name = 'default' if name is None else name

        if isinstance(model, six.string_types):
            if model == 'PL' or model == 'PowerLaw':
                model = sherpa.models.PowLaw1D('powlaw1d.' + name)
                model.gamma = 2
                model.ref = 1e9
                model.ampl = 1e-20
            else:
                raise ValueError("Undefined model string: {}".format(model))

        if not isinstance(model, sherpa.models.ArithmeticModel):
            raise ValueError("Only sherpa models are supported")

        self._model = model

    @property
    def statistic(self):
        """Statistic to be used in the fit"""
        return self._stat

    @statistic.setter
    def statistic(self, stat):
        """Set Statistic to be used in the fit

        Parameters
        ----------
        stat : `~sherpa.stats.Stat`, str
            Statistic
        """
        import sherpa.stats as s

        if isinstance(stat, six.string_types):
            if stat == 'cash':
                stat = s.Cash()
            elif stat == 'wstat':
                stat = s.WStat()
            else:
                raise ValueError("Undefined stat string: {}".format(stat))

        if not isinstance(stat, s.Stat):
            raise ValueError("Only sherpa statistics are supported")

        self._stat = stat

    @property
    def energy_threshold_low(self):
        """
        Low energy threshold of the spectral fit
        """
        return self._thres_lo

    @energy_threshold_low.setter
    def energy_threshold_low(self, energy):
        """
        Low energy threshold setter

        Parameters
        ----------
        energy : `~gammapy.utils.energy.Energy`, str
            Low energy threshold
        """
        self._thres_lo = Energy(energy)

    @property
    def energy_threshold_high(self):
        """
       High energy threshold of the spectral fit
        """
        return self._thres_hi

    @energy_threshold_high.setter
    def energy_threshold_high(self, energy):
        """
        High energy threshold setter

        Parameters
        ----------
        energy : `~gammapy.utils.energy.Energy`, str
            High energy threshold
        """
        self._thres_hi = Energy(energy)

    @property
    def pha_list(self):
        """Comma-separate list of PHA files"""
        ret = ''
        for p in self.pha:
            ret += str(p) + ","

        return ret

    def info(self):
        """Print some basic info"""
        ss = 'Model\n'
        ss += str(self.model)
        ss += '\nEnergy Range\n'
        ss += str(self.energy_threshold_low) + ' - ' + str(
                self.energy_threshold_high)
        return ss

    def run(self, method='hspec'):
        if method == 'hspec':
            self._run_hspec_fit()
        elif method == 'sherpa':
            self._run_sherpa_fit()
        else:
            raise ValueError('Undefined fitting method')

    def _run_hspec_fit(self):
        """Run the gammapy.hspec fit
        """

        log.info("Starting HSPEC")
        import sherpa.astro.ui as sau
        from ..hspec import wstat

        sau.set_conf_opt("max_rstat", 100)

        thres_lo = self.energy_threshold_low.to('keV').value
        thres_hi = self.energy_threshold_high.to('keV').value
        sau.freeze(self.model.ref)

        list_data = []
        for pha in self.pha:
            datid = pha.parts[-1][7:12]
            sau.load_data(datid, str(pha))
            sau.notice_id(datid, thres_lo, thres_hi)
            sau.set_source(datid, self.model)
            list_data.append(datid)

        wstat.wfit(list_data)

    def _run_sherpa_fit(self):
        """Plain sherpa fit not using the session object
        """
        from sherpa.astro import datastack
        log.info("Starting SHERPA")
        log.info(self.info())
        ds = datastack.DataStack()
        ds.load_pha(self.pha_list)
        ds.set_source(self.model)
        thres_lo = self.energy_threshold_low.to('keV').value
        thres_hi = self.energy_threshold_high.to('keV').value
        ds.notice(thres_lo, thres_hi)
        datastack.set_stat(self.statistic)
        ds.fit()
        datastack.covar()
        covar = datastack.get_covar_results()
        efilter = datastack.get_filter()

        # First go on calculation flux points following
        # http://cxc.harvard.edu/sherpa/faq/phot_plot.html
        # This should be split out and improved
        xx = datastack.get_fit_plot().dataplot.x
        dd = datastack.get_fit_plot().dataplot.y
        ee = datastack.get_fit_plot().dataplot.yerr
        mm = datastack.get_fit_plot().modelplot.y
        src = datastack.get_source()(xx)
        points = dd / mm * src
        errors = ee / mm * src
        flux_graph = dict(energy=xx, flux=points, flux_err_hi=errors,
                          flux_err_lo=errors)

        from gammapy.spectrum.results import SpectrumFitResult
        self.result = SpectrumFitResult.from_sherpa(covar, efilter, self.model,
                                                    flux_graph)
        ds.clear_stack()
        ds.clear_models()

    def apply_containment(self, fit):
        """Apply correction factor for PSF containment in ON region"""
        cont = self.get_containment()
        raise NotImplementedError

    def get_containment(self):
        """Calculate PSF correction factor for containment in ON region"""
        # TODO: do something useful here
        return 1


def run_spectral_fit_using_config(config):
    """
    Run a 1D spectral analysis using a config dict

    This function is called by the ``gammapy-spectrum`` command line tool.
    See :ref:`spectrum_command_line_tool`.

    Parameters
    ----------
    config : dict
        Config dict

    Returns
    -------
    fit : `~gammapy.spectrum.SpectrumFit`
        Fit instance
    """
    log.info("\nStarting analysis {}".format(config['general']['outdir']))
    outdir = make_path(config['general']['outdir'])

    if config['general']['create_ogip']:
        analysis = SpectrumAnalysis.from_config(config)
        analysis.write_ogip_data(str(outdir / 'ogip_data'))
        total_stats = analysis.total_spectrum.spectrum_stats
        print(total_stats.to_table())

    method = config['general']['run_fit']
    if method is not False:
        fit = SpectrumFit.from_config(config)
        fit.model = config['model']['type']
        fit.energy_threshold_low = Energy(config['model']['threshold_low'])
        fit.energy_threshold_high = Energy(config['model']['threshold_high'])
        fit.info()
        fit.run(method=method)
        log.info("\n\n*** Fit Result ***\n\n{}\n\n\n".format(fit.result.to_table()))
        fit.result.to_yaml(str(outdir / 'fit_result.yaml'))
        return fit



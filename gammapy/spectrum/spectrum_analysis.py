# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (print_function)
import logging
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy.extern import six
from gammapy.extern.pathlib import Path
from gammapy.image import ExclusionMask
from gammapy.region import SkyCircleRegion, find_reflected_regions
from ..background import ring_area_factor
from ..data import CountsSpectrum
from ..obs import DataStore
from ..spectrum import EnergyBounds, Energy
from ..utils.scripts import get_parser, set_up_logging_from_args

__all__ = [
    'SpectrumAnalysis',
    'SpectrumObservation',
    'run_spectrum_analysis_using_config',
    'run_spectrum_analysis_using_configfile',
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

    run_spectrum_analysis_using_configfile(args.config_file)


class SpectrumAnalysis(object):
    """Class for 1D spectrum fitting

    Parameters
    ----------
    datastore : str
        Name of a `~gammapy.obs.Data store`
    obs : list, str
        List of observations or file containing such a list
    on_region : `gammapy.region.SkyCircleRegion`
        Circular region to extract on counts
    exclusion : `~gammapy.image.ExclusionMask`
        Exclusion regions
    bkg_method : dict, optional
        Background method including necessary parameters
    nobs : int
        number of observations to process
    ebounds : `~gammapy.spectrum.EnergyBounds`, optional
        Reconstructed energy binning definition

    """

    def __init__(self, datastore, obs, on_region, exclusion, bkg_method=None,
                 nobs=-1, ebounds=None):

        self.on_region = on_region
        self.store = DataStore.from_name(datastore)
        self.exclusion = exclusion
        if ebounds is None:
            ebounds = EnergyBounds.equal_log_spacing(0.1, 10, 20, 'TeV')
        if bkg_method is None:
            bkg_method = dict(type='no method')

        if isinstance(obs, six.string_types):
            obs = np.loadtxt(obs, dtype=np.int)

        self._observations = []
        for i, val in enumerate(obs):
            try:
                temp = SpectrumObservation(val, self.store, on_region,
                                           bkg_method, ebounds, exclusion)
            except IndexError:
                log.warn(
                    'Observation {} not in store {}'.format(val, datastore))
                nobs = nobs + 1
                continue
            self._observations.append(temp)
            if i == nobs - 1:
                break

        if len(self.observations) == 0:
            raise ValueError("No valid observations found")

    @property
    def observations(self):
        """List of all observations belonging to the analysis
        """
        return self._observations

    @property
    def reflected_regions(self, **kwargs):
        """List of dicts containing information about the reflected regions
        for each observation
        """
        retval = list([])
        for obs in self.observations:
            reflected = obs.make_reflected_regions(**kwargs)
            val = dict(obs=obs.obs, pointing=obs.pointing, region=reflected)
            retval.append(val)
        return retval

    def info(selfs):
        """Print some information
        """
        pass

    def write_ogip_data(self, dir=None):
        """Create OGIP files

        Parameters
        ----------
        dir : str (optional)
            write directory
        """
        if dir is None:
            dir = 'ogip_data'

        for obs in self.observations:
            obs.write_all_ogip_data(dir)
            log.info('Creating OGIP data for run{}'.format(obs.obs))

    def run_hspec_fit(self, model, thres_low, thres_high):
        """Run the gammapy.hspec fit

        Parameters
        ----------
        model : str
            Sherpa model
        thres_high : `~gammapy.spectrum.Energy`
            Upper threshold of the spectral fit
        thres_low : `~gammapy.spectrum.Energy`
            Lower threshold of the spectral fit
        """

        log.info("Starting HSPEC")
        import sherpa.astro.ui as sau
        from ..hspec import wstat
        from sherpa.models import PowLaw1D

        if model == 'PL':
            p1 = PowLaw1D('p1')
            p1.gamma = 2.2
            p1.ref = 1e9
            p1.ampl = 6e-19
        else:
            raise ValueError('Desired Model is not defined')

        thres = thres_low.to('keV').value
        emax = thres_high.to('keV').value

        sau.freeze(p1.ref)
        sau.set_conf_opt("max_rstat", 100)

        list_data = []
        for obs in self.observations:
            datid = obs.phafile.parts[-1][7:12]
            sau.load_data(datid, str(obs.phafile))
            sau.notice_id(datid, thres, emax)
            sau.set_source(datid, p1)
            list_data.append(datid)
        wstat.wfit(list_data)
        sau.covar()
        fit_val = sau.get_covar_results()
        fit_attrs = ('parnames', 'parvals', 'parmins', 'parmaxes')
        fit = dict((attr, getattr(fit_val, attr)) for attr in fit_attrs)
        fit = self.apply_containment(fit)
        sau.clean()
        self.fit = fit

    def apply_containment(self, fit):
        """Apply correction factor for PSF containment in ON region"""
        cont = self.get_containment()
        fit['containment'] = cont
        fit['parvals'] = list(fit['parvals'])
        fit['parvals'][1] = fit['parvals'][1] * cont
        return fit

    def get_containment(self):
        """Calculate PSF correction factor for containment in ON region"""
        return 1


class SpectrumObservation(object):
    """Helper class for 1D region based spectral analysis

    This class handles the spectrum fit for one observation/run

    TODO: Link to example

    Parameters
    ----------
    obs : int
        Observation ID, runnumber
    store : `~gammapy.obs.DataStore
        Data Store
    ebounds : `~gammapy.spectrum.EnergyBounds`
        Reconstructed energy binning definition
    on_region : `gammapy.region.SkyCircleRegion`
        Circular region to extract on counts
    bkg_method : dict
        Background method including necessary parameters

    """

    def __init__(self, obs, store, on_region, bkg_method, ebounds, exclusion):

        # Raises Error if obs is not available
        store.filename(obs, 'events')
        self.obs = obs
        self.store = store
        self.on_region = on_region
        self.bkg_method = bkg_method
        self.ebounds = ebounds
        self.exclusion = exclusion
        self._event_list = None
        self.pha = None
        self.bkg = None
        self.arf = None
        self.rmf = None
        self.phafile = None

    @property
    def event_list(self):
        """`~gammapy.data.EventList` corresponding to the observation
        """
        if self._event_list is None:
            self._event_list = self.store.load(obs_id=self.obs,
                                              filetype='events')
        return self._event_list

    @property
    def pointing(self):
        """`~astropy.coordinates.SkyCoord corresponding to the obs position
        """
        return self.event_list.pointing_radec

    @property
    def offset(self):
        """`~astropy.coordinates.Angle corresponding to the obs offset
        """
        return self.pointing.separation(self.on_region.pos)

    def make_on_vector(self):
        """Create ON vector

        Returns
        -------
        on_vec : `gammapy.data.CountsSpectrum`
            Counts spectrum inside the ON region
        """
        on_list = self.event_list.select_circular_region(self.on_region)
        on_vec = CountsSpectrum.from_eventlist(on_list, self.ebounds)
        self.pha = on_vec
        return on_vec

    def make_reflected_regions(self, **kwargs):
        """Create reflected off regions

        Returns
        -------
        off_region : `~gammapy.region.SkyRegionList`
            Reflected regions

        kwargs are forwarded to gammapy.region.find_reflected_regions
        """
        off_region = find_reflected_regions(self.on_region, self.pointing,
                                            self.exclusion, **kwargs)
        return off_region

    def make_off_vector(self):
        """Create off vector

        Returns
        -------
        on_vec : `gammapy.data.CountsSpectrum`
            Counts spectrum inside the OFF region
        """
        if self.bkg_method['type'] == "ring":
            # TODO put in utils once there is a SkyRingRegion
            center = self.on_region.pos
            radius = self.on_region.radius
            inner = self.bkg_method['inner_radius']
            outer = self.bkg_method['outer_radius']
            off_list = self.event_list.select_sky_ring(center, inner, outer)
            alpha = ring_area_factor(radius.deg, inner.deg, outer.deg)
        elif self.bkg_method['type'] == "reflected":
            kwargs = self.bkg_method.copy()
            kwargs.pop('type')
            off = self.make_reflected_regions(**kwargs)
            off_list = self.event_list.select_circular_region(off)
            alpha = len(off)
        else:
            raise ValueError("Undefined background method: {}".format(
                self.bkg_method['type']))

        off_vec = CountsSpectrum.from_eventlist(off_list, self.ebounds)
        off_vec.backscal = alpha
        self.bkg = off_vec
        return off_vec

    def make_arf(self):
        """Create effective area vector correct energy binning

        Returns
        -------
        arf : `~gammapy.irf.EffectiveAreaTable`
             effective area vector
        """
        aeff2d = self.store.load(obs_id=self.obs, filetype='aeff')
        arf_vec = aeff2d.to_effective_area_table(self.offset)
        self.arf = arf_vec
        return arf_vec

    def make_rmf(self):
        """Create energy disperion matrix in correct energy binning

        Returns
        -------
        rmf : `~gammapy.irf.EnergyDispersion`
            energy dispersion matrix
        """
        edisp2d = self.store.load(obs_id=self.obs, filetype='edisp')
        rmf_mat = edisp2d.to_energy_dispersion(self.offset,
                                               e_reco=self.ebounds)
        self.rmf = rmf_mat
        return rmf_mat

    def write_ogip(self, phafile=None, bkgfile=None, rmffile=None, arffile=None,
                   outdir=None, clobber=True):
        """Write OGIP files

        Only those objects are written have been created with the appropriate
        functions before

        Parameters
        ----------
        phafile : str
            PHA filename
        bkgfile : str
            BKG filename
        arffile : str
            ARF filename
        rmffile : str
            RMF : filename
        outdir : None
            directory to write the files to
        clobber : bool
            Overwrite
        """

        if outdir is None:
            outdir = "ogip_data"

        basedir = Path(outdir)
        basedir.mkdir(exist_ok=True)

        if arffile is None:
            arffile = basedir / "arf_run{}.fits".format(self.obs)
        if rmffile is None:
            rmffile = basedir / "rmf_run{}.fits".format(self.obs)
        if phafile is None:
            phafile = basedir / "pha_run{}.pha".format(self.obs)
        if bkgfile is None:
            bkgfile = basedir / "bkg_run{}.pha".format(self.obs)

        self.phafile = phafile

        if self.pha is not None:
            self.pha.write(str(phafile), bkg=str(bkgfile), arf=str(arffile),
                           rmf=str(rmffile), clobber=clobber)
        if self.bkg is not None:
            self.bkg.write(str(bkgfile), clobber=clobber)
        if self.arf is not None:
            self.arf.write(str(arffile), energy_unit='keV', effarea_unit='cm2',
                           clobber=clobber)
        if self.rmf is not None:
            self.rmf.write(str(rmffile), energy_unit='keV', clobber=clobber)

    def write_all_ogip_data(self, dir):
        """Perform all step to provide the OGIP data for a sherpa fit

        Parameters
        ----------
        dir : str
            Directory to write to
        """
        self.make_on_vector()
        self.make_off_vector()
        self.make_arf()
        self.make_rmf()
        self.write_ogip(outdir=dir)

    def _check_binning(self):
        """Check that ARF and RMF binnings are compatible
        """
        pass


def run_spectrum_analysis_using_configfile(configfile):
    """Wrapper function to run a 1D spectral analysis using a config file

    Parameters
    ----------
    configfile : str
        Config file in YAML format

    Returns
    -------
    analysis : `~gammapy.spectrum.SpectrumAnalysis`
        Spectrum analysis instance
    """
    import yaml
    log.info('Reading {}'.format(configfile))
    with open(configfile) as fh:
        config = yaml.safe_load(fh)

    analysis = run_spectrum_analysis_using_config(config)
    return analysis


def run_spectrum_analysis_using_config(config):
    """Wrapper function to run a 1D spectral analysis using a config dict

    Parameters
    ----------
    config : dict
        config

    Returns
    -------
    analysis : `~gammapy.spectrum.SpectrumAnalysis`
        Spectrum analysis instance
    """

    # Observations
    obs = config['general']['runlist']
    store = config['general']['datastore']
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
    log.debug('Binning: {}'.format(ebounds))

    # ON region
    radius = Angle(config['on_region']['radius'])
    x = config['on_region']['center_x']
    y = config['on_region']['center_y']
    frame = config['on_region']['system']
    center = SkyCoord(x, y, frame=frame)
    on_region = SkyCircleRegion(center, radius)

    # OFF region
    off_type = config['off_region']['type']
    if off_type == 'ring':
        irad = Angle(config['off_region']['inner_radius'])
        orad = Angle(config['off_region']['outer_radius'])
        bkg_method = dict(type='ring', inner_radius=irad,
                          outer_radius=orad)
    elif off_type == 'reflected':
        bkg_method = dict(type='reflected')

    # Exclusion
    excl_file = config['excluded_regions']['file']
    exclusion = ExclusionMask.from_fits(excl_file)

    analysis = SpectrumAnalysis(datastore=store, obs=obs, on_region=on_region,
                                bkg_method=bkg_method, exclusion=exclusion,
                                nobs=nobs, ebounds=ebounds)
    
    if config['general']['create_ogip']:
        outdir = config['general']['outdir']
        analysis.write_ogip_data(outdir)

    if config['general']['run_fit']:
        model = config['model']['type']
        thres_low = Energy(config['model']['threshold_low'])
        thres_high = Energy(config['model']['threshold_high'])
        fit = analysis.run_hspec_fit(model, thres_low, thres_high)

    return analysis

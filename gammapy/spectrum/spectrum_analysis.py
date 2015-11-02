# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (print_function)
from ..utils.scripts import get_parser, set_up_logging_from_args
from ..obs import DataStore, ObservationTable
from ..irf import EnergyDispersion, EnergyDispersion2D
from ..irf import EffectiveAreaTable, EffectiveAreaTable2D
from ..data import CountsSpectrum, EventList
from ..spectrum import EnergyBounds, Energy
from ..background import ring_area_factor
from astropy.coordinates import Angle, SkyCoord
from astropy.extern import six
from astropy.io import fits
import logging
import numpy as np
import os

__all__ = ['SpectrumAnalysis', 'SpectrumObservation']

log = logging.getLogger(__name__)


def main(args=None):
    parser = get_parser(SpectrumAnalysis)
    parser.add_argument('config_file', type=str,
                        help='Config file in YAML format')
    parser.add_argument("-l", "--loglevel", default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help="Set the logging level")

    args = parser.parse_args(args)
    set_up_logging_from_args(args)
    analysis = SpectrumAnalysis.from_yaml(args.config_file)
    analysis.run()


class SpectrumAnalysis(object):
    """Perform a 1D spectrum fit
    """

    def __init__(self, config):
        self.config = config
        vals = config['general']['runlist']
        if isinstance(vals, six.string_types):
            vals = np.loadtxt(vals, dtype=np.int)
        # TODO: add while loop
        try:
            self.obs = vals[0]
            _process_config(self)
        except IOError:
            self.obs = vals[1]
            _process_config(self)

        log.info('Creating analysis ' + self.outdir)
        self.observations = []
        nruns = self.config['general']['nruns'] - 1
        for i, obs in enumerate(vals):
            try:
                val = SpectrumObservation(obs, config)
            except IOError:
                log.warn('Run ' + str(obs) + ' does not exist - skipping')
                nruns = nruns + 1
                continue
            self.observations.append(val)
            if i == nruns:
                break

    @classmethod
    def from_yaml(cls, filename):
        """Read config from YAML file."""
        import yaml
        log.info('Reading {}'.format(filename))
        with open(filename) as fh:
            config = yaml.safe_load(fh)
        return cls(config)

    def info(self):
        log.info(self.store.info())
        log.info('ON region\nCenter: {0}\nRadius: {1}'.format(
            self.target, self.radius))
        log.info('OFF region\nType: {0}\nInner Radius: {1}\nOuter Radius: {2}'.format(
            'Ring', self.irad, self.orad))

    def run(self):
        """Run analysis as specified in the config"""
        if self.config['general']['create_ogip']:
            self.make_ogip()
        if self.config['general']['run_fit']:
            fit = self.run_fit()
            return fit

    def make_ogip(self):
        """Create OGIP files"""
        for obs in self.observations:
            obs.make_ogip()
            log.info('Creating OGIP data for run{}'.format(obs.obs))

    def run_fit(self):
        """Run the gammapy.hspec fit"""
        log.info("Starting HSPEC")
        import sherpa.astro.ui as sau
        from ..hspec import wstat
        from sherpa.models import PowLaw1D

        if (self.model == 'PL'):
            p1 = PowLaw1D('p1')
            p1.gamma = 2.2
            p1.ref = 1e9
            p1.ampl = 6e-19
        else:
            raise ValueError('Desired Model is not defined')

        sau.freeze(p1.ref)
        sau.set_conf_opt("max_rstat", 100)

        list_data = []
        for obs in self.observations:
            runfile = obs.phafile
            datid = runfile.split('/')[2][7:12]
            sau.load_data(datid, runfile)
            sau.notice_id(datid, self.thres, self.emax)
            sau.set_source(datid, p1)
            list_data.append(datid)
        wstat.wfit(list_data)
        sau.covar()
        fit_val = sau.get_covar_results()
        fit_attrs = ('parnames', 'parvals', 'parmins', 'parmaxes')
        fit = dict((attr, getattr(fit_val, attr)) for attr in fit_attrs)
        fit = self.apply_containment(fit)
        sau.clean()
        return fit

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
    """1D region based spectral analysis observation.

    This class handles the spectrum fit for one observation/run
    """

    def __init__(self, obs, config):
        self.config = config
        self.obs = obs
        _process_config(self)

    def make_ogip(self):
        """Write OGIP files needed for the sherpa fit

        The 'clobber' kwarg is set to true in this function
        """
        self._prepare_ogip()
        clobber = True
        self.pha.write(self.phafile, bkg=self.bkgfile, arf=self.arffile,
                       rmf=self.rmffile, clobber=clobber)
        self.bkg.write(self.bkgfile, clobber=clobber)
        self.arf.write(self.arffile, energy_unit='keV',
                       effarea_unit='cm2', clobber=clobber)
        self.rmf.write(self.rmffile, energy_unit='keV', clobber=clobber)

    def _prepare_ogip(self):
        """Dummy function to process IRFs and event list
        """
        self.make_on_vector()
        self.make_off_vector()
        self.make_arf()
        self.make_rmf()

    def make_on_vector(self):
        """Make ON `~gammapy.data.CountsSpectrum`
        """
        on_list = self.event_list.select_sky_cone(self.target, self.radius)
        on_vec = CountsSpectrum.from_eventlist(on_list, self.ebounds)
        self.pha = on_vec

    def make_off_vector(self):
        """Make OFF `~gammapy.data.CountsSpectrum`
        """
        if self.off_type == "ring":
            off_list = self.event_list.select_sky_ring(
                self.target, self.irad, self.orad)
        elif self.off_type == "reflected":
            off_list = self.event_list.select_reflected_regions(
                self.target, self.radius, self.exclusion)
        else:
            raise ValueError("Undefined background method: {}".format(self.off_type))
            
        off_vec = CountsSpectrum.from_eventlist(off_list, self.ebounds)
        off_vec.backscal = self.alpha
        self.bkg = off_vec

    def make_arf(self):
        """Make `~gammapy.irf.EffectiveAreaTable`
        """
        aeff2D_file = self.store.filename(self.obs, 'aeff')
        aeff2D = EffectiveAreaTable2D.read(aeff2D_file)
        self.arf = aeff2D.to_effective_area_table(self.offset)

    def make_rmf(self):
        """Make `~gammapy.irf.EnergyDispersion`
        """
        edisp2D_file = self.store.filename(self.obs, 'edisp')
        edisp2D = EnergyDispersion2D.read(edisp2D_file)
        self.rmf = edisp2D.to_energy_dispersion(self.offset, e_reco=self.ebounds)

    def _check_binning(self):
        """Check that ARF and RMF binnings are compatible
        """
        pass


def _process_config(object):
    """Helper function to process the config file
    """

    # Data
    storename = object.config['general']['datastore']
    object.store = DataStore.from_name(storename)
    object.outdir = object.config['general']['outdir']
    basename = object.outdir + "/ogip_data"

    # TODO: use Path here (see Developer HOWTO entry why / how).
    if not os.path.isdir(object.outdir):
        os.mkdir(object.outdir)
        os.mkdir(basename)
    object.arffile = basename + "/arf_run" + str(object.obs) + ".fits"
    object.rmffile = basename + "/rmf_run" + str(object.obs) + ".fits"
    object.phafile = basename + "/pha_run" + str(object.obs) + ".pha"
    object.bkgfile = basename + "/bkg_run" + str(object.obs) + ".pha"

    # Target
    x = Angle(object.config['on_region']['center_x'])
    y = Angle(object.config['on_region']['center_y'])
    frame = object.config['on_region']['system']
    object.target = SkyCoord(x, y, frame=frame)

    # Pointing
    event_list = object.store.load(obs_id = object.obs, filetype ='events')
    object.event_list = event_list
    object.pointing = object.event_list.pointing_radec
    object.offset = object.target.separation(object.pointing)

    # Excluded regions
    excl_file = object.config['excluded_regions']['file']
    object.exclusion = fits.open(excl_file)[0]

    # Binning
    sec = object.config['binning']
    if sec['equal_log_spacing']:
        emin = Energy(sec['emin'])
        emax = Energy(sec['emax'])
        nbins = sec['nbins']
        object.ebounds = EnergyBounds.equal_log_spacing(
            emin, emax, nbins)
    else:
        if sec[binning] is None:
            raise ValueError("No binning specified")
    log.debug('Binning: {}'.format(object.ebounds))

    # ON/OFF Region
    val = object.config['on_region']['radius']
    object.radius = Angle(val)

    object.off_type = object.config['off_region']['type']
    if object.off_type == 'ring':
        ival = object.config['off_region']['inner_radius']
        oval = object.config['off_region']['outer_radius']
        object.irad = Angle(ival)
        object.orad = Angle(oval)
        object.alpha = ring_area_factor(object.radius, object.irad, object.orad).value
    elif object.off_type == 'reflected':
        pass
    
    # Spectral fit
    object.model = object.config['model']['type']
    val = object.config['model']['threshold_low']
    val2 = object.config['model']['threshold_high']
    threshold = Energy(val)
    threshold2 = Energy(val2)
    object.thres = threshold.to('keV').value
    object.emax = threshold2.to('keV').value

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
        vals = config['general']['observations']
        if isinstance(vals, basestring):
            vals = np.loadtxt(vals, dtype=np.int)
        self.obs = vals[0]
        _process_config(self)
        self.info()
        self.observations = []
        for obs in vals:
            val = SpectrumObservation(obs, config)
            self.observations.append(val)

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
            print(fit)

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
            datid = runfile.split('/')[1][7:12]
            sau.load_data(datid, runfile)
            sau.notice_id(datid, self.thres, self.emax)
            sau.set_source(datid, p1)
            list_data.append(datid)
        wstat.wfit(list_data)
        fit_val = sau.get_fit_results()
        fit_attrs = ('parnames', 'parvals')
        fit = dict((attr, getattr(fit_val, attr)) for attr in fit_attrs)
        return fit

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
                       effarea_unit='cm2',          clobber=clobber)
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
        off_list = self.event_list.select_sky_ring(self.target, self.irad, self.orad)
        off_vec = CountsSpectrum.from_eventlist(off_list, self.ebounds)
        off_vec.backscal = self.alpha
        self.bkg = off_vec

    def make_arf(self):
        """Make `~gammapy.irf.EffectiveAreaTable`
        """
        aeff2D_file = self.store.filename(self.obs, 'effective area')
        aeff2D = EffectiveAreaTable2D.read(aeff2D_file)
        self.arf = aeff2D.to_effective_area_table(self.offset)

    def make_rmf(self):
        """Make `~gammapy.irf.EnergyDispersion`
        """
        edisp2D_file = self.store.filename(self.obs, 'energy dispersion')
        edisp2D = EnergyDispersion2D.read(edisp2D_file)
        self.rmf = edisp2D.to_energy_dispersion(self.ebounds, self.offset)

    def _check_binning(self):
        """Check that ARF and RMF binnings are compatible
        """
        pass


def _process_config(object):
    """Helper function to process the config file
    """

    # Data
    storedir = object.config['general']['datastore']
    object.store = DataStore(dir=storedir)
    outdir = object.config['general']['outdir']
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    object.arffile = outdir + "/arf_run" + str(object.obs) + ".fits"
    object.rmffile = outdir + "/rmf_run" + str(object.obs) + ".fits"
    object.phafile = outdir + "/pha_run" + str(object.obs) + ".pha"
    object.bkgfile = outdir + "/bkg_run" + str(object.obs) + ".pha"

    # Target
    x = Angle(object.config['on_region']['center_x'])
    y = Angle(object.config['on_region']['center_y'])
    frame = object.config['on_region']['system']
    object.target = SkyCoord(x, y, frame=frame)

    # Pointing
    event_list_file = object.store.filename(object.obs, 'events')
    event_list = EventList.read(event_list_file, hdu=1)
    object.event_list = event_list
    object.pointing = object.event_list.pointing_radec
    object.offset = object.target.separation(object.pointing)

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
    ival = object.config['off_region']['inner_radius']
    oval = object.config['off_region']['outer_radius']
    object.irad = Angle(ival)
    object.orad = Angle(oval)
    object.alpha = ring_area_factor(object.radius, object.irad, object.orad).value

    # Spectral fit
    object.model = object.config['model']['type']
    val = object.config['model']['threshold_low']
    val2 = object.config['model']['threshold_high']
    threshold = Energy(val)
    threshold2 = Energy(val2)
    object.thres = threshold.to('keV').value
    object.emax = threshold2.to('keV').value

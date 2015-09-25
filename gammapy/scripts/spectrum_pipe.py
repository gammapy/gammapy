# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (print_function)
import logging
from ..utils.scripts import get_parser, set_up_logging_from_args
from ..obs import DataStore, ObservationTable
from ..irf import EnergyDispersion, EnergyDispersion2D
from ..irf import EffectiveAreaTable, EffectiveAreaTable2D
from ..data import CountsSpectrum, EventList
from ..spectrum import EnergyBounds
from astropy.coordinates import Angle, SkyCoord

__all__ = ['GammapySpectrumAnalysis', 'GammapySpectrumObservation']

log = logging.getLogger(__name__)


def main(args=None):
    parser = get_parser(GammapySpectrumAnalysis)
    parser.add_argument('config_file', type=str,
                        help='Config file in YAML format')
    parser.add_argument("-l", "--loglevel", default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help="Set the logging level")
    args = parser.parse_args(args)
    set_up_logging_from_args(args)
    analysis = GammapySpectrumAnalysis.from_yaml(args.config_file)
    #optional
    analysis.make_ogip()
    #analysis.run_fait()

class GammapySpectrumAnalysis(object):
    """Entrance points for gammapy-spectrum-pipe
    """
    def __init__(self, config):
        self.config = config
        self.obs = config['general']['observations'][0]
        _process_config(self)
        self.info()
        self.observations = []
        for obs in config['general']['observations']:
            val = GammapySpectrumObservation(obs, config)
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
            self.target,self.radius))
        log.info('OFF region\nType: {0}\nInner Radius: {1}\nOuter Radius: {2}'.format(
            'Ring',self.irad,self.orad))

    def make_ogip(self):
        for obs in self.observations:
            log.info('Creating OGIP data for run{}'.format(obs.obs))
            obs.make_ogip()

    def run_fit(self):
        pass


class GammapySpectrumObservation(object):
    """Gammapy 1D region based spectral analysis.
    """

    def __init__(self, obs, config):
        self.config = config
        self.obs = obs
        _process_config(self)

    def make_ogip(self):
        """Run analysis chain."""
        self._prepare_ogip()

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

    def write_ogip(self):
        """Write OGIP files needed for the sherpa fit
        """
        
        arffile = "arf_run"+self.obs+".fits"
        rmffile = "rmf_run"+self.obs+".fits"
        phafile = "pha_run"+self.obs+".pha"
        bkgfile = "bkg_run"+self.obs+".pha"
            
    def _check_binning(self):
        """Check that ARF and RMF binnings are compatible
        """
        pass

    def set_model(self):
        """Specify the fit model
        """
        pass

    def run_hspec(self):
        """Run HSPEC analysis
        """
        pass


def _process_config(object):
    """Helper function to process the config file
    """

    storedir = object.config['general']['datastore']
    object.store = DataStore(dir=storedir)
    unit = object.config['on_region']['unit']
    x = Angle(object.config['on_region']['center_x'], unit)
    y = Angle(object.config['on_region']['center_y'], unit)
    frame = object.config['on_region']['system']
    object.target = SkyCoord(x,y,frame = frame)
        
    sec = object.config['binning']
    if sec['equal_log_spacing']:
        object.ebounds = EnergyBounds.equal_log_spacing(
            sec['emin'],sec['emax'],sec['nbins'],sec['unit'])
    else:
        if sec[binning] is None:
            raise ValueError("No binning specified")
    log.debug('Binning: {}'.format(object.ebounds))
            
    event_list_file = object.store.filename(object.obs, 'events')
    event_list = EventList.read(event_list_file, hdu=1)
    object.event_list = event_list
    object.pointing = object.event_list.pointing_radec
    object.offset = object.target.separation(object.pointing)

    val = object.config['on_region']['radius']
    object.radius = Angle(val, 'deg')

    ival = object.config['off_region']['inner_radius']
    oval = object.config['off_region']['outer_radius']
    object.irad = Angle(ival, 'deg')
    object.orad = Angle(oval, 'deg')

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

__all__ = ['GammapySpectrumAnalysis']

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
    analysis.run()


class GammapySpectrumAnalysis(object):
    """Gammapy 1D region based spectral analysis.
    """

    def __init__(self, config):
        self.config = config
        self._process_config()
        self.event_list = []
        self.aeff2D_table = []
        self.edisp2D_table = []
        self.pha = []
        self.bkg = []
        self.arf = []
        self.rmf = []

    @classmethod
    def from_yaml(cls, filename):
        """Read config from YAML file."""
        import yaml
        log.info('Reading {}'.format(filename))
        with open(filename) as fh:
            config = yaml.safe_load(fh)
        return cls(config)

    def run(self):
        """Run analysis chain."""
        log.info('Running analysis ...')
        self.get_fits_data()
        self.make_on_vector()
        self.make_off_vector()
        self.make_arf()
        self.make_rmf()

    def _process_config(self):
        storedir = self.config['general']['datastore']
        self.store = DataStore(dir=storedir)
        unit = self.config['on_region']['unit']
        x = Angle(self.config['on_region']['center_x'], unit)
        y = Angle(self.config['on_region']['center_y'], unit)
        frame = self.config['on_region']['system']
        self.target = SkyCoord(x,y,frame = frame)
        
        sec = self.config['binning']
        if sec['equal_log_spacing']:
            self.ebounds = EnergyBounds.equal_log_spacing(
                sec['emin'],sec['emax'],sec['nbins'],sec['unit'])
        else:
            if sec[binning] is None:
                raise ValueError("No binning specified")
        log.debug('Binning: {}'.format(self.ebounds))

    def get_fits_data(self):
        """Find FITS files according to observations specified in the config file
        """
        log.info('Retrieving data from datastore.')
        for obs in self.config['general']['observations']:
            event_list_file = self.store.filename(obs, 'events')
            event_list = EventList.read(event_list_file, hdu=1)
            self.event_list.append(event_list)
            
            aeff2D_table_file = self.store.filename(obs, 'effective area')
            aeff2D_table = EffectiveAreaTable2D.read(aeff2D_table_file)
            self.aeff2D_table.append(aeff2D_table)
            
            edisp2D_table_file = self.store.filename(obs, 'energy dispersion')
            edisp2D_table = EnergyDispersion2D.read(edisp2D_table_file)
            self.edisp2D_table.append(edisp2D_table)
            
    def make_on_vector(self):
        """Make ON `~gammapy.data.CountsSpectrum`
        """
        val = self.config['on_region']['radius']
        radius = Angle(val, 'deg')
        log.info('Creating circular ON region\n'
                 'Center: {0}\nRadius: {1}'.format(self.target,radius))

        for list in self.event_list:
            on_list = list.select_sky_cone(self.target, radius)
            on_vec = CountsSpectrum.from_eventlist(on_list, self.ebounds)
            self.pha.append(on_vec)

    def make_off_vector(self):
        """Make OFF `~gammapy.data.CountsSpectrum`
        """
        ival = self.config['off_region']['inner_radius']
        oval = self.config['off_region']['outer_radius']
        irad = Angle(ival, 'deg')
        orad = Angle(oval, 'deg')
        log.info('Creating OFF region\nType: {0}\nInner Radius: {1}\nOuter'
                 ' Radius: {2}'.format('Ring',irad,orad))

        for list in self.event_list:
            off_list = list.select_sky_ring(self.target, irad, orad)
            off_vec = CountsSpectrum.from_eventlist(off_list, self.ebounds)
            self.bkg.append(off_vec)
    
    def make_arf(self):
        """Make `~gammapy.irf.EffectiveAreaTable`
        """
        for list, aeff2D in zip(self.event_list, self.aeff2D_table):
            pointing = list.pointing_radec
            offset = self.target.separation(pointing)
            self.arf.append(aeff2D.to_effective_area_table(offset))

    def make_rmf(self):
        """Make `~gammapy.irf.EnergyDispersion`
        """

        for list, edisp2D in zip(self.event_list, self.edisp2D_table):
            pointing = list.pointing_radec
            offset = self.target.separation(pointing)
            self.rmf.append(edisp2D.to_energy_dispersion(self.ebounds, offset))

    def write_ogip(self):
        """Write OGIP files needed for the sherpa fit
        """
        for obs in self.config['general']['observations']:
            arffile = "arf_run"+obs+".fits"
            rmffile = "rmf_run"+obs+".fits"
            phafile = "pha_run"+obs+".pha"
            bkgfile = "bkg_run"+obs+".pha"
            
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

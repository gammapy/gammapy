# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (print_function)
import logging
from ..utils.scripts import get_parser, set_up_logging_from_args
from ..obs import DataStore, ObservationTable

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
        self.store = DataStore(dir=config['general']['datastore'])
        self.event_list = []
        self.aeff2d_table = []
        self.edisp2d_table = []

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
        self.get_fits_files()
        
    def get_fits_files(self):
        """Find FITS files according to observations specified in the config file
        """
        log.info('Retrieving data from datastore ...')
        for obs in self.config['general']['observations']:
            self.event_list.append(self.store.filename(obs, 'events'))
            self.aeff2d_table.append(self.store.filename(obs, 'effective area'))
            self.edisp2d_table.append(self.store.filename(obs, 'energy dispersion'))
            
        
    def make_on_vector(self):
        """Make ON `~gammapy.data.CountsSpectrum`
        """

        pass

    def make_off_vector(self):
        """Make ON `~gammapy.data.CountsSpectrum`
        """
        pass
    
    def make_arf(self):
        """Make `~gammapy.irf.EffectiveAreaTable`
        """

        pass

    def make_rmf(self):
        """Make `~gammapy.irf.EnergyDispersion`
        """

        pass

    def write_ogip(self):
        """Write OGIP files needed for the sherpa fit
        """
        pass
        
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

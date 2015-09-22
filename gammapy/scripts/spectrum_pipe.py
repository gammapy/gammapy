# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (print_function)
import logging
from ..utils.scripts import get_parser, set_up_logging_from_args

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
        print(self.config['general']['outdir'])


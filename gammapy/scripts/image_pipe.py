# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging
from ..utils.scripts import get_parser

__all__ = ['ImageAnalysis']

log = logging.getLogger(__name__)


def image_pipe_main(args=None):
    parser = get_parser(ImageAnalysis)
    parser.add_argument('config_file', type=str,
                        help='Config file in YAML format')
    # TODO: add option to dump the default config file
    args = parser.parse_args(args)
    analysis = ImageAnalysis.from_yaml(args.config_file)
    analysis.run()


class ImageAnalysis(object):
    """Gammapy 2D image based analysis.
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
        print(self.config)

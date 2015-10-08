# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (print_function)
from gammapy.spectrum.spectrum_analysis import SpectrumAnalysis
from ..utils.scripts import get_parser, set_up_logging_from_args
import logging
import numpy as np
__all__ = ['SpectrumPipe']

log = logging.getLogger(__name__)

def main(args=None):
    parser = get_parser(SpectrumPipe)
    parser.add_argument('config_file', type=str,
                        help='Config file in YAML format')
    parser.add_argument("-l", "--loglevel", default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help="Set the logging level")

    args = parser.parse_args(args)
    set_up_logging_from_args(args)
    specpipe = SpectrumPipe.from_yaml(args.config_file)
    specpipe.run()


class SpectrumPipe(object):
    """Gammapy Spectrum Pipe class"""
    def __init__(self, config):
        self.config = config
        fit_config_file = config['general']['spectrum_fit_config_file']
        fit_config = read_yaml(fit_config_file)
        sec = self.config['sources']
        sources = sec.keys()
        self.analysis = []
        for target in sources:
            vals = sec[target]
            fit_config['general']['outdir'] = target
            fit_config['general']['runlist'] = vals['runlist']
            fit_config['on_region']['center_x'] = vals['target_ra']
            fit_config['on_region']['center_y'] = vals['target_dec']
            analysis = SpectrumAnalysis(fit_config)
            write_yaml(fit_config, target+"/"+target)
            self.analysis.append(analysis)

            
    @classmethod
    def from_yaml(cls, filename):
        config = read_yaml(filename)
        return cls(config)

    def run(self):
        """Run Spectrum Analysis Pipe"""
        self.result = []
        for ana in self.analysis:
            log.info("Starting Analysis for target "+ana.outdir)
            fit = ana.run()
            self.result.append(fit)
        self.print_result()

    def print_result(self):
        """Print Fit Results"""
        print('\n------------------------------')
        for res, ana in zip(self.result, self.analysis):
            gamma = res['parvals'][0]
            gamma_err = res['parmaxes'][0]
            norm = res['parvals'][1]*1e9
            norm_err = res['parmaxes'][1]*1e9
            print('\n')            
            print(ana.outdir)
            print('Gamma     : {0:.3f} +/- {1:.3f}'.format(gamma, gamma_err))
            print('Flux@1TeV : {0:.3e} +/- {1:.3e}'.format(norm, norm_err))

# TODO -> utils
def read_yaml(filename):
    """Read config from YAML file."""
    import yaml
    log.info('Reading {}'.format(filename))
    with open(filename) as fh:
        config = yaml.safe_load(fh)
    return config

def write_yaml(config, filename):
    """Write YAML config file"""
    import yaml
    filename = filename+'.yaml'
    log.info('Writing {}'.format(filename))
    with open(filename, 'w') as outfile:
        outfile.write( yaml.dump(config, default_flow_style=False))
    
def to_json(data):
    import json

    

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (print_function)
from gammapy.spectrum.spectrum_analysis import SpectrumAnalysis
from ..utils.scripts import get_parser, set_up_logging_from_args, write_yaml, read_yaml
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
            try:
                fit_config['on_region']['radius'] = vals['on_radius']
            except KeyError:
                pass
            analysis = SpectrumAnalysis(fit_config)
            write_yaml(fit_config, target+"/"+target, log)
            self.analysis.append(analysis)

            
    @classmethod
    def from_yaml(cls, filename):
        config = read_yaml(filename, log)
        return cls(config)

    def run(self):
        """Run Spectrum Analysis Pipe"""
        self.result = dict()
        for ana in self.analysis:
            log.info("Starting Analysis for target "+ana.outdir)
            fit = ana.run()
            self.result[ana.outdir] = fit
        self.print_result()
        ref_vals_file = self.config['general']['reference_values'] 
        if ref_vals_file is not None:
            self.make_comparison_plot(ref_vals_file)
        
    def print_result(self):
        """Print Fit Results"""
        print('\n------------------------------')
        for target, res in self.result.iteritems():
            gamma = res['parvals'][0]
            gamma_err = res['parmaxes'][0]
            norm = res['parvals'][1]*1e9
            norm_err = res['parmaxes'][1]*1e9
            print('\n')            
            print('Target     : {}'.format(target))
            print('Gamma      : {0:.3f} +/- {1:.3f}'.format(gamma, gamma_err))
            print('Flux@1TeV  : {0:.3e} +/- {1:.3e}'.format(norm, norm_err))
            print('Containment: {0:.1%}'.format(res['containment']))
        print('\n------------------------------\n')

    def make_comparison_plot(self, filename):
        """Create comparison plot

        This function takes some reference values for the spectrum
        pipeline and create a plot that visualizes the deviation
        of the pipeline results to the reference values
        """
        import matplotlib.pyplot as plt


        ref = read_yaml(filename, log)
        labels = []
        g_diff = []
        g_diff_err = []
        for target, res in self.result.iteritems():
            try:
                sec = ref[target]
            except KeyError:
                log.warn('No reference values found in {0} for '
                         'analysis {1}'.format(filename, target))

            else:
              labels.append(target)
              g_ref =  float(sec['index'].split()[0])
              g_ref_err =  float(sec['index'].split()[1])
              g_act = res['parvals'][0]
              g_act_err = res['parvals'][1]
              
              g_diff.append(g_ref-g_act)
              g_diff_err.append(np.sqrt(g_ref_err**2+g_act_err**2))

        x = np.arange(len(labels))
        plt.errorbar(x,g_diff, yerr=g_diff_err, fmt='b.')
        plt.xticks(x, labels, size='small', rotation=45)
        min = -1
        max = len(x)
        plt.xlim(min, max)
        plt.ylabel('Index - Reference Value')
        plt.plot(np.linspace(min,max,10), np.zeros(10), 'r-')
        val = filename.split('.')[0]
        plt.savefig('comparison_to_{}.png'.format(val))

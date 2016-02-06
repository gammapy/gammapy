# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (print_function)

import logging
import numpy as np

from gammapy.spectrum import run_spectrum_extraction_using_config
from gammapy.spectrum.spectrum_fit import run_spectrum_fit_using_config
from gammapy.utils.scripts import read_yaml, recursive_merge_dicts, make_path, \
    write_yaml

__all__ = ['SpectrumPipe']

log = logging.getLogger(__name__)


class SpectrumPipe(object):
    """Gammapy Spectrum Pipe class

    Parameters
    ----------
    config : list
        List of configuration files
    """

    def __init__(self, config):
        self.config = config

    @classmethod
    def from_configfile(cls, filename, auto_outdir=True):
        """Create `~gammapy.script.SpectrumPipe` from config file

        Parameters
        ----------
        filename : str
            YAML configfile
        auto_outdir : bool [True]
            Set outdir explicitly for every analysis
        """
        config = read_yaml(filename, log)
        return cls.from_config(config, auto_outdir=auto_outdir)

    @classmethod
    def from_config(cls, config, auto_outdir=True):
        """Create `~gammapy.script.SpectrumPipe` from config dict

        Parameters
        ----------
        config : dict
            config dict
        auto_outdir : bool [True]
            Set outdir explicitly for every analysis
        """
        base_config = config.pop('base_config')
        analist = list([])

        for analysis in config.keys():
            log.info("Generating config for analysis {}".format(analysis))
            anaconf = base_config.copy()
            temp = config[analysis]
            anaconf = recursive_merge_dicts(anaconf, temp)
            if auto_outdir:
                anaconf['extraction']['results']['outdir'] = analysis
                anaconf['fit']['outdir'] = analysis

            analist.append(anaconf)

        return cls(analist)

    def write_configs(self):
        """Write analysis configs to disc"""
        for conf in self.config:
            outdir = make_path(conf['fit']['outdir'])
            outdir.mkdir(exist_ok=True)
            outfile = outdir / 'config.yaml'
            write_yaml(conf, str(outfile), logger=log)

    def info(self):
        """
        Basic information about the analysis pipeline
        """
        raise NotImplementedError

    def run(self):
        """Run spectrum pipe"""
        for conf in self.config:
            run_spectrum_analysis_using_config(conf)


def run_spectrum_analysis_using_config(config):
    """Run entire specturm analysis

    This function simply calls
    * :func:`gammapy.spectrum.spectrum_extraction.run_spectrum_extraction_using_config()
    * :func:`gammapy.spectrum.spectrum_fit.run_spectrum_fit_using_config()

    Parameters
    ----------
    config : dict
       config dict with keys 'extraction' and 'fit'

    Returns
    -------
    fit : `~gammapy.spectrum.spectrum_fit.SpectrumFit`
        Spectrum fit instance
    analysis : `~gammapy.spectrum.spectrum_extraction.SpectrumExtraction`
        Spectrum extraction analysis instance
    """

    analysis = run_spectrum_extraction_using_config(config)
    fit = run_spectrum_fit_using_config(config)

    #Todo: add utility to not have to specify same outdir twice
    #Todo: add utility to write only one outputfile

    return fit, analysis
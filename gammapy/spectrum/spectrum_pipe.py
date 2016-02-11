# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (print_function)

import copy
import logging
import numpy as np

from ..spectrum import run_spectrum_extraction_using_config
from ..spectrum.spectrum_fit import run_spectrum_fit_using_config
from ..utils.scripts import read_yaml, recursive_merge_dicts, make_path, \
    write_yaml

__all__ = ['SpectrumPipe']

log = logging.getLogger(__name__)


class SpectrumPipe(object):
    """Gammapy Spectrum Pipe class
    """

    def __init__(self):
        pass


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

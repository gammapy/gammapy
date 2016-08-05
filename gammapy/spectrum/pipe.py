# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (print_function)

import logging


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

    raise NotImplementedError

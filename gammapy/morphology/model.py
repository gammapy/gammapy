# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility funtions for reading / writing
model parameters to JSON files.

At the moment you can have any number of Gaussians
"""
from __future__ import print_function, division
import logging
import json

__all__ = ['GaussCatalog', 'make_test_model', 'read_json']

class GaussCatalog(dict):
    """Multi-Gauss catalog utils."""

    def __init__(self, source):
        if isinstance(source, dict):
            # Assume source is a dict with correct format
            self.pars = source
        elif isinstance(source, str):
            # Assume it is a JSON filename
            fh = open(source)
            self.pars = json.load(fh)
            fh.close()
        else:
            logging.error('Unknown source: {0}'.format(source))

    def set(self):
        ' + '.join(['gauss2d.' + name for name in source_names])
        pass


def make_test_model(nsources=100, npix=500, ampl=100, fwhm=30):
    """Create a model of several Gaussian sources.
    """
    from numpy.random import random
    from sherpa.astro.ui import set_source
    from morphology.utils import _set, _name
    model = ' + '.join([_name(ii) for ii in range(nsources)])
    set_source(model)
    for ii in range(nsources):
        _set(_name(ii), 'xpos', npix * random())
        _set(_name(ii), 'ypos', npix * random())
        _set(_name(ii), 'ampl', ampl * random())
        _set(_name(ii), 'fwhm', fwhm * random())


def read_json(filename):
    from sherpa.astro.ui import set_source
    morphology.utils.read_json(filename, set_source)

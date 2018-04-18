# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from . import SkyModelMapEvaluator
from ..stats import cash
from ..utils.fitting import fit_minuit


class CubeFit(object):
    """Perform 3D likelihood fit

    This is the first go at such a class. It's geared to the
    `~gammapy.spectrum.SpectrumFit` class which does the 1D spectrum fit.

    Parameters
    ----------
    counts : `~gammapy.maps.WcsNDMap`
        Counts cube
    exposure : `~gammapy.maps.WcsNDMap`
        Exposure cube
    model : `~gammapy.cube.SkyModel`
        Fit model
    """
    def __init__(self, model, counts, exposure):
        self.model = model
        self.counts = counts
        self.exposure = exposure
        self._init_evaluator()
        
        self._npred = None
        self._stat = None
        self._minuit = None

    @property
    def npred(self):
        """Predicted counts cube"""
        return self._npred

    @property
    def stat(self):
        """Fit statistic per bin"""
        return self._stat

    @property
    def minuit(self):
        """`~iminuit.Minuit` object"""
        return self._npred

    def _init_evaluator(self):
        """Initialize SkyModelEvaluator"""
        self.evaluator = SkyModelMapEvaluator(self.model,
                                              self.exposure)
        
    def compute_npred(self):
        """Compute predicted counts"""
        self._npred = self.evaluator.compute_npred()

    def compute_stat(self):
        """Compute fit statistic per bin"""
        self._stat = cash(
            n_on=self.counts.data,
            mu_on=self.npred
        )

    def total_stat(self, parameters):
        """Likelihood for a given set of model parameters"""
        self.model.parameters = parameters
        self.compute_npred()
        self.compute_stat()
        total_stat = np.sum(self.stat, dtype=np.float64)
        return total_stat

    def fit(self):
        """Run the fit"""
        parameters, minuit = fit_minuit(parameters=self.model.parameters,
                                        function=self.total_stat)
        self._minuit = minuit

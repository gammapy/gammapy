# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from .models import SkyModelMapEvaluator
from ..stats import cash
from ..utils.fitting import fit_iminuit

__all__ = [
    'SkyModelMapFit',
]


class SkyModelMapFit(object):
    """Perform sky model likelihood fit on maps.

    This is the first go at such a class. It's geared to the
    `~gammapy.spectrum.SpectrumFit` class which does the 1D spectrum fit.

    Parameters
    ----------
    model : `~gammapy.cube.SkyModel`
        Fit model
    counts : `~gammapy.maps.WcsNDMap`
        Counts cube
    exposure : `~gammapy.maps.WcsNDMap`
        Exposure cube
    psf : `~gammapy.cube.PSFKernel`
        PSF kernel
    background : `~gammapy.maps.WcsNDMap`
        Background Cube
    """

    def __init__(self, model, counts, exposure, psf=None, background=None):
        self.model = model
        self.counts = counts
        self.exposure = exposure
        self.psf = psf
        self.background = background

        self._npred = None
        self._stat = None
        self._minuit = None

        self._init_evaluator()

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
        return self._minuit

    def _init_evaluator(self):
        """Initialize SkyModelEvaluator"""
        self.evaluator = SkyModelMapEvaluator(sky_model=self.model,
                                              exposure=self.exposure,
                                              psf=self.psf,
                                              background=self.background)

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
        parameters, minuit = fit_iminuit(parameters=self.model.parameters,
                                         function=self.total_stat)
        self._minuit = minuit

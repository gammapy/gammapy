# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from .models import MapEvaluator
from ..stats import cash
from ..utils.fitting import fit_iminuit

__all__ = [
    'MapFit',
]


class MapFit(object):
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
    background : `~gammapy.maps.WcsNDMap`
        Background Cube
    psf : `~gammapy.cube.PSFKernel`
        PSF kernel
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion
    """

    def __init__(self, model, counts, exposure, background=None, psf=None, edisp=None):
        self.model = model
        self.counts = counts
        self.exposure = exposure
        self.background = background
        self.psf = psf
        self.edisp = edisp

        self._npred = None
        self._stat = None
        self._minuit = None

        self.evaluator = MapEvaluator(
            model=self.model,
            exposure=self.exposure,
            background=self.background,
            psf=self.psf,
            edisp=self.edisp,
        )

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
        return np.sum(self.stat, dtype=np.float64)

    def fit(self, opts_minuit=None):
        """Run the fit

        Parameters
        ----------
        opts_minuit : dict (optional)
            Options passed to `iminuit.Minuit` constructor
        """
        minuit = fit_iminuit(parameters=self.model.parameters,
                             function=self.total_stat,
                             opts_minuit=opts_minuit)
        self._minuit = minuit

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""This module contains helper classes for the sherpa backend of
`~gammapy.spectrum.SpectrumFit.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from sherpa.models import ArithmeticModel, modelCacher1d, Parameter
from sherpa.stats import Likelihood

__all__ = [
    'SherpaModel',
    'SherpaStat',
    'SherpaExponentialCutoffPowerLaw',
]


class SherpaModel(ArithmeticModel):
    """Dummy sherpa model for the `~gammapy.spectrum.SpectrumFit`

    Parameters
    ----------
    fit : `~gammapy.spectrum.SpectrumFit`
        Fit instance
    """

    def __init__(self, fit):
        self.fit = fit

        sherpa_name = 'sherpa_model'
        par_list = list()
        for par in self.fit.model.parameters.parameters:
            sherpa_par = par.to_sherpa(modelname='source')
            # setattr(self, name, sherpa_par)
            par_list.append(sherpa_par)

        if fit.stat != 'wstat' and self.fit.background_model is not None:
            for par in self.fit.background_model.parameters.parameters:
                sherpa_par = par.to_sherpa(modelname='background')
                # setattr(self, name, sherpa_par)
                par_list.append(sherpa_par)

        ArithmeticModel.__init__(self, sherpa_name, par_list)
        self._use_caching = True
        self.cache = 10

    @modelCacher1d
    def calc(self, p, x, xhi=None):
        # Adjust model parameters
        n_src = len(self.fit.model.parameters.parameters)
        for idx, par in enumerate(p):
            # Special case background model
            if idx >= n_src:
                self.fit.background_model.parameters.parameters[idx - n_src].value = par
            else:
                self.fit.model.parameters.parameters[idx].value = par

        self.fit.predict_counts()
        # Return ones since sherpa does some check on the shape
        return np.ones_like(self.fit.obs_list[0].e_reco)


class SherpaStat(Likelihood):
    """Dummy sherpa stat for the `~gammapy.spectrum.SpectrumFit`

    Parameters
    ----------
    fit : `~gammapy.spectrum.SpectrumFit`
        Fit instance
    """

    def __init__(self, fit):
        sherpa_name = 'sherpa_stat'
        self.fit = fit
        Likelihood.__init__(self, sherpa_name)

    def _calc(self, data, model, *args, **kwargs):
        self.fit.calc_statval()
        # Sum likelihood over all observations
        total_stat = np.sum([np.sum(v) for v in self.fit.statval], dtype=np.float64)
        # sherpa return pattern: total stat, fvec
        return total_stat, None


class SherpaExponentialCutoffPowerLaw(ArithmeticModel):
    """Exponential CutoffPowerLaw.

    Note that the cutoff is given in units '1/TeV' in order to bring the Sherpa
    optimizers into a valid range. All other parameters still have units 'keV'
    and 'cm2'.
    """

    def __init__(self, name='ecpl'):
        self.gamma = Parameter(name, 'gamma', 2, min=-10, max=10)
        self.ref = Parameter(name, 'ref', 1, frozen=True)
        self.ampl = Parameter(name, 'ampl', 1, min=0)
        self.cutoff = Parameter(name, 'cutoff', 1, min=0, units='1/TeV')
        ArithmeticModel.__init__(self, name, (self.gamma, self.ref, self.ampl,
                                              self.cutoff))
        self._use_caching = True
        self.cache = 10

    @modelCacher1d
    def calc(self, p, x, xhi=None):
        from .models import ExponentialCutoffPowerLaw
        kev_to_tev = 1e-9
        model = ExponentialCutoffPowerLaw(index=p[0],
                                          reference=p[1],
                                          amplitude=p[2],
                                          lambda_=p[3] * kev_to_tev)
        if xhi is None:
            val = model(x)
        else:
            val = model.integral(x, xhi, intervals=True)

        return val

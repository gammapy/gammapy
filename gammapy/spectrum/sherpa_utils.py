# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""This module contains helper classes for the sherpa backend of
`~gammapy.spectrum.SpectrumFit"""

from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
from sherpa.models import ArithmeticModel, Parameter, modelCacher1d
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
        # TODO: add Parameter and ParameterList class
        self.fit = fit
        source_pars = OrderedDict(**self.fit.model.parameters)
        # "Freeze reference by removing it from fit parameters
        # TODO: Add proper
        source_pars.pop('reference')
        self.sorted_pars = source_pars
        
        # Add background model pars if needed
        if self.fit.background_model is not None:
            bkg_pars = OrderedDict(**self.fit.background_model.parameters)
            bkg_pars.pop('reference')
            for par in bkg_pars:
                # TODO: Find better way do mark background parameters
                self.sorted_pars['bkg_{}'.format(par)] = bkg_pars[par]

        sherpa_name = 'sherpa_model'
        par_list = list()
        for name, par in self.sorted_pars.items():
            sherpa_par = Parameter(sherpa_name,
                                   name,
                                   par.value,
                                   units=str(par.unit))
            setattr(self, name, sherpa_par)
            par_list.append(sherpa_par)

        ArithmeticModel.__init__(self, sherpa_name, par_list)
        self._use_caching = True
        self.cache = 10

    @modelCacher1d
    def calc(self, p, x, xhi=None):
        # Adjust model parameters
        for par, parval in zip(self.sorted_pars, p):
            par_unit = self.sorted_pars[par].unit
            if 'bkg' in par:
                self.fit.background_model.parameters[par[4:]] = parval * par_unit
            else:
                self.fit.model.parameters[par] = parval * par_unit

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
        total_stat = np.sum(self.fit.statval)
        # sherpa return pattern: total stat, fvec
        return total_stat, None


class SherpaExponentialCutoffPowerLaw(ArithmeticModel):
    """Exponential CutoffPowerLaw
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

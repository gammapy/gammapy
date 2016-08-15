# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Sherpa spectral models
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from sherpa.models import ArithmeticModel, Parameter, modelCacher1d

__all__ = [
    'SherpaExponentialCutoffPowerLaw',
]

# Partly copied from https://github.com/zblz/naima/blob/master/naima/sherpa_models.py#L33


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

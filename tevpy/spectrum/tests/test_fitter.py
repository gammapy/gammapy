# Licensed under a 3-clause BSD style license - see LICENSE.rst
'''
from spec.data import FluxPoints
from spec.models import PowerLaw
from spec.fitter import Fitter

def fit_crab_with_pl():
    """Fit a constant to some test data"""
    data = FluxPoints.from_ascii('input/crab_hess_spec.txt')
    print data
    model = PowerLaw()
    print model
    fitter = Fitter(data, model)
    print fitter
    fitter.fit()
    print fitter
'''

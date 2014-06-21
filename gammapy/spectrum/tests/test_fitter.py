# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.tests.helper import pytest
from ..flux_point import compute_differential_flux_points
from ..models import PowerLaw
from ..fitter import Fitter


@pytest.mark.xfail
def fit_crab_with_pl():
    """Fit a constant to some test data"""
    data = FluxPoints.from_ascii('input/crab_hess_spec.txt')
    model = PowerLaw()
    fitter = Fitter(data, model)
    fitter.fit()

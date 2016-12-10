# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest
from ...spectrum import Fitter
from ...spectrum.models import PowerLaw


@pytest.mark.xfail
def fit_crab_with_pl():
    """Fit a constant to some test data"""
    # TODO: FluxPoints class currently doesn't exist
    data = FluxPoints.from_ascii('input/crab_hess_spec.txt')
    model = PowerLaw()
    fitter = Fitter(data, model)
    fitter.fit()

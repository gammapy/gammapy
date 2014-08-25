# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.tests.helper import pytest
from ...spectrum import (compute_differential_flux_points,
                         Fitter,
                         )
from ...spectrum.models import PowerLaw


@pytest.mark.xfail
def fit_crab_with_pl():
    """Fit a constant to some test data"""
    # TODO: FluxPoints class currently doesn't exist
    data = FluxPoints.from_ascii('input/crab_hess_spec.txt')
    model = PowerLaw()
    fitter = Fitter(data, model)
    fitter.fit()

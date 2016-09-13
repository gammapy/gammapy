# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.units import Quantity
from astropy.tests.helper import assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data
from ...datasets import gammapy_extra
from ...spectrum import (
    calculate_flux_point_binning,
    SpectrumObservation,
)


@requires_data('gammapy-extra')
@requires_dependency('scipy')
def test_flux_points_binning():
    phafile = gammapy_extra.filename("datasets/hess-crab4_pha/pha_obs23523.fits")
    obs = SpectrumObservation.read(phafile)
    energy_binning = calculate_flux_point_binning(obs_list=[obs], min_signif=3)
    assert_quantity_allclose(energy_binning[5], Quantity(1.668, 'TeV'), rtol=1e-3)

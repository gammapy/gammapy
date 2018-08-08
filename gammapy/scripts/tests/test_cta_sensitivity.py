# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
import astropy.units as u
from gammapy.utils.testing import requires_data, requires_dependency
from gammapy.stats import significance_on_off
from ...irf.io import CTAPerf
from ..cta_sensitivity import SensitivityEstimator


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_cta_sensitivity():
    """Run sensitivity estimation for one CTA IRF example."""
    # TODO: change the test case to something simple that's easy to understand?
    # E.g. a step function in AEFF and a very small Gaussian EDISP?
    filename = '$GAMMAPY_EXTRA/datasets/cta/perf_prod2/point_like_non_smoothed/South_5h.fits.gz'
    irf = CTAPerf.read(filename)

    sens = SensitivityEstimator(irf=irf, livetime=5.0 * u.h)
    sens.run()
    table = sens.diff_sensi_table

    assert len(table) == 21

    # Assert on relevant values in three energy bins
    # TODO: add asserts on other quantities: excess, exposure
    assert_allclose(table['ENERGY'][0], 0.015848932787775993)
    assert_allclose(table['FLUX'][0], 1.265689381204479e-10)

    assert_allclose(table['ENERGY'][9], 1)
    assert_allclose(table['FLUX'][9], 4.2875940141105596e-13)

    assert_allclose(table['ENERGY'][20], 158.4893035888672)
    assert_allclose(table['FLUX'][20], 9.048305001092968e-12)


# TODO: fix this test
@pytest.mark.xfail
@requires_dependency('scipy')
def test_cta_min_gamma():
    """Run sensitivity estimation for one CTA IRF example."""

    sens = SensitivityEstimator(
        irf=None,
        livetime=5.0 * u.h,
        gamma_min=100
    )
    e = sens.get_excess([0, 0, 0, 0])
    assert_allclose([100, 100, 100, 100], e)

    # Assert on a few rows of the result table
    assert len(table) == 21

    assert_allclose(
        table['ENERGY'][[0, 9, 20]],
        [0.0158489, 1.0, 158.489],
        rtol=0.01,
    )
    assert_allclose(
        table['FLUX'][[0, 9, 20]],
        [1.223534e-10, 4.272442e-13, 9.047706e-12],
        rtol=0.01,
    )

# TODO: fix this test
@pytest.mark.xfail
@requires_dependency('scipy')
def test_cta_correct_sigma():
    """Run sensitivity estimation for one CTA IRF example."""

    sens = SensitivityEstimator(
        irf=None,
        livetime=5.0 * u.h,
        gamma_min=10,
        sigma=10.0
    )
    excess = sens.get_excess([1200])
    off = 1200 * 5
    on = excess + 1200
    sigma = significance_on_off(on, off, alpha=0.2)
    assert_almost_equal(sigma, 10, decimal=1)
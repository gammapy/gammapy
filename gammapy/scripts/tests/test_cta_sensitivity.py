# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose, assert_almost_equal
from gammapy.utils.testing import requires_data, requires_dependency
from gammapy.stats import significance_on_off
import astropy.units as u
from ..cta_irf import CTAPerf
from ..cta_sensitivity import SensitivityEstimator


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_cta_sensitivity():
    """Run sensitivity estimation for one CTA IRF example."""
    filename = '$GAMMAPY_EXTRA/datasets/cta/perf_prod2/point_like_non_smoothed/South_5h.fits.gz'
    irf = CTAPerf.read(filename)

    sens = SensitivityEstimator(irf=irf, livetime=5.0 * u.h)
    sens.run()
    table = sens.diff_sensi_table

    # Assert on diff flux at 1 TeV
    assert_allclose(table[9].data[1], 6.8452201495e-13, rtol=0.01)


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

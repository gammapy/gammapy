# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from gammapy.utils.testing import requires_data, requires_dependency
import astropy.units as u
from ..cta_irf import CTAPerf
from ..cta_sensitivity import SensitivityEstimator


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_cta_sensitivity():
    """Run sensitivity estimation for one CTA IRF example."""
    filename = '$GAMMAPY_EXTRA/datasets/cta/perf_prod2/point_like_non_smoothed/South_5h.fits.gz'
    irf = CTAPerf.read(filename)

    sens = SensitivityEstimator(irf=irf, livetime=5.0*u.Unit('h'))
    sens.run()
    table = sens.diff_sensi_table

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

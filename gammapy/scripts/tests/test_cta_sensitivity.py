# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from gammapy.utils.testing import requires_data, requires_dependency
from ..cta_irf import CTAPerf
from ..cta_sensitivity import SensiEstimator


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_cta_sensitivity():
    """Run sensitivity estimation for one CTA IRF example."""
    filename = '$GAMMAPY_EXTRA/datasets/cta/perf_prod2/North_0.5h/irf_file.fits.gz'
    irf = CTAPerf.read(filename)

    sens = SensiEstimator(irf=irf, livetime='0.5 hour')
    sens.run()
    table = sens.diff_sensi_table

    # Assert on diff flux at 1 TeV
    assert_allclose(table[9], 6.8452201495e-13, rtol=0.01)

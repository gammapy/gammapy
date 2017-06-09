# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.tests.helper import assert_quantity_allclose
from numpy.testing import assert_allclose
from astropy.units import Quantity
from gammapy.utils.testing import requires_data, requires_dependency
from .. import cta_sensitivity
from .. import CTAPerf

@requires_dependency('scipy')
@requires_data('gammapy-extra')

def test_cta_sensitivity():
    """Test that CTA sensitivity can be well evaluated."""

    irffile = '$GAMMAPY_EXTRA/datasets/cta/perf_prod2/North_0.5h/irf_file.fits.gz'
    ctairf = CTAPerf.read(irffile)
    livetime = 0.5
    sens = cta_sensitivity.SensiEstimator(irfobject=ctairf,livetime=livetime)
    sens.run()

    tt = sens.print_results()
    #actual = sens.diff_sens[np.where(np.abs(sens.energy.value - 1.) < 0.2)][0].value
    actual = tt[9]  #Diff flux at 1 TeV

    true_value = 6.8452201495e-13 #Diff flux at 1 TeV
    assert_allclose(actual, true_value, rtol=0.01)

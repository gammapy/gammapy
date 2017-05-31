# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.tests.helper import assert_quantity_allclose
from numpy.testing import assert_allclose
from astropy.units import Quantity
from gammapy.utils.testing import requires_data, requires_dependency
from .. import cta_sensitivity


@requires_dependency('scipy')
@requires_data('gammapy-extra')

def test_cta_sensitivity():
    """Test that CTA sensitivity can be well evaluated."""

    irffile = '$GAMMAPY_EXTRA/datasets/cta/CTA-Performance-North-20170327/CTA-Performance-North-20deg-average-30m_20170327.fits'
    livetime = 0.5
    sens = cta_sensitivity.cta_sensi_estim(irffile=irffile,livetime=livetime)
    sens.run()

    true_value =6.3596014545e-12
    actual = sens.diff_sens[np.where(np.abs(sens.energy.value - 1.) < 0.2)][0].value

    assert_allclose(actual, true_value, rtol=0.01)

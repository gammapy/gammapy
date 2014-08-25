# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ...irf import EnergyDispersion


@pytest.mark.xfail
def test_EnergyDispersion():
    edisp = EnergyDispersion()
    pdf = edisp(3, 4)
    assert_allclose(pdf, 42)

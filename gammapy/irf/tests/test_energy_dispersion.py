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



def test_EnergyDispersion2D():

    # Read test effective area file
    effarea = EffectiveAreaTable2D.from_fits(
        load_aeff2D_fits_table())

    effarea.interpolation_method = method

    # Check that nodes are evaluated correctly
    e_node = 42
    off_node = 3
    offset = effarea.offset[off_node]
    energy = effarea.energy[e_node]
    actual = effarea.evaluate(offset, energy)
    desired = effarea.eff_area[off_node, e_node]
    assert_allclose(actual, desired)


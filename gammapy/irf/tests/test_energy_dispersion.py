# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from numpy.testing import assert_allclose, assert_equal
from astropy.tests.helper import pytest
from ...irf import EnergyDispersion, EnergyDispersion2D
from ...datasets import load_edisp2D_fits_table

@pytest.mark.xfail
def test_EnergyDispersion():
    edisp = EnergyDispersion()
    pdf = edisp(3, 4)
    assert_allclose(pdf, 42)

def test_EnergyDispersion2D():

    # Read test effective area file
    edisp = EnergyDispersion2D.from_fits(
        load_edisp2D_fits_table())

    # Check that nodes are evaluated correctly
    e_node = 12
    off_node = 3
    m_node = 5
    offset = edisp.offset[off_node]
    energy = edisp.energy.log_centers[e_node]
    migra = edisp.migra[m_node]
    actual = edisp.evaluate(offset, energy, migra)
    desired = edisp.dispersion[off_node, m_node, e_node]
    assert_allclose(actual, desired, rtol=1e-06)


    # Check that values between node make sense
    #THINK ABOUT WHAT MAKES SENSE
    energy2 = edisp.energy.log_centers[e_node + 1]
    upper = edisp.evaluate(offset, energy, migra)
    lower = edisp.evaluate(offset, energy2, migra)
    e_val = (energy + energy2) / 2
    actual = edisp.evaluate(offset, e_val, migra)
    #assert_equal(lower > actual and actual > upper, True)


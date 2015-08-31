# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from numpy.testing import assert_allclose, assert_equal
from astropy.tests.helper import pytest, remote_data
from ...irf import EnergyDispersion, EnergyDispersion2D
from ...datasets import get_path


try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False



@pytest.mark.xfail
def test_EnergyDispersion():
    edisp = EnergyDispersion()
    pdf = edisp(3, 4)
    assert_allclose(pdf, 42)

@pytest.mark.skipif('not HAS_SCIPY')
@remote_data
def test_EnergyDispersion2D():

    filename = get_path("../test_datasets/irf/hess/pa/hess_edisp_2d_023523.fits.gz",
                        location='remote')

    # Read test effective area file
    edisp = EnergyDispersion2D.read(filename)

    # Check that nodes are evaluated correctly
    e_node = 12
    off_node = 3
    m_node = 5
    offset = edisp.offset[off_node]
    energy = edisp.energy[e_node]
    migra = edisp.migra[m_node]
    actual = edisp.evaluate(offset, energy, migra)
    desired = edisp.dispersion[off_node, m_node, e_node]
    assert_allclose(actual, desired, rtol=1e-06)


    # Check that values between node make sense
    #THINK ABOUT WHAT MAKES SENSE
    energy2 = edisp.energy[e_node + 1]
    upper = edisp.evaluate(offset, energy, migra)
    lower = edisp.evaluate(offset, energy2, migra)
    e_val = (energy + energy2) / 2
    actual = edisp.evaluate(offset, e_val, migra)
    #assert_equal(lower > actual and actual > upper, True)


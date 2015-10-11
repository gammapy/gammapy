# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.utils.data import get_pkg_data_filename
from astropy.tests.helper import pytest
from astropy.io import fits

from ...irf import EffectiveAreaTable2D, EffectiveAreaTable, abramowski_effective_area
from ...datasets import load_arf_fits_table, load_aeff2D_fits_table

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def test_abramowski_effective_area():
    energy = Quantity(100, 'GeV')
    area_ref = Quantity(1.65469579e+07, 'cm^2')

    area = abramowski_effective_area(energy, 'HESS')
    assert_allclose(area, area_ref)
    assert area.unit == area_ref.unit

    energy = Quantity([0.1, 2], 'TeV')
    area_ref = Quantity([1.65469579e+07, 1.46451957e+09], 'cm^2')

    area = abramowski_effective_area(energy, 'HESS')
    assert_allclose(area, area_ref)
    assert area.unit == area_ref.unit


@pytest.mark.skipif('not HAS_SCIPY')
def test_EffectiveAreaTable():
    filename = get_pkg_data_filename('data/arf_info.txt')
    info_str = open(filename, 'r').read()
    arf = EffectiveAreaTable.from_fits(load_arf_fits_table())
    print(arf.info())
    assert arf.info() == info_str


def test_EffectiveAreaTable_write(tmpdir):
    irf = EffectiveAreaTable.from_fits(load_arf_fits_table())
    filename = str(tmpdir.join('effarea_test.fits'))
    irf.write(filename)

    # Verify checksum
    hdu_list = fits.open(filename)
    # TODO: replace this assert with something else.
    # For unknown reasons this verify_checksum fails non-deterministically
    # see e.g. https://travis-ci.org/gammapy/gammapy/jobs/31056341#L1162
    # assert hdu_list[1].verify_checksum() == 1
    assert len(hdu_list) == 2


INTERPOLATION_METHODS = ['linear', 'spline']


@pytest.mark.parametrize(('method'), INTERPOLATION_METHODS)
@pytest.mark.skipif('not HAS_SCIPY')
def test_EffectiveAreaTable2D(method):
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

    # Check that values between node make sense
    energy2 = effarea.energy[e_node + 1]
    upper = effarea.evaluate(offset, energy)
    lower = effarea.evaluate(offset, energy2)
    e_val = (energy + energy2) / 2
    actual = effarea.evaluate(offset, e_val)
    assert_equal(lower > actual and actual > upper, True)

    # Test evaluate function (return shape)
    # Case 0; offset = scalar, energy = scalar, done

    # Case 1: offset = scalar, energy = None
    offset = Angle(0.234, 'degree')
    actual = effarea.evaluate(offset=offset).shape
    desired = effarea.energy.shape
    assert_equal(actual, desired)

    # Case 2: offset = scalar, energy = 1Darray
    offset = Angle(0.564, 'degree')
    nbins = 42
    energy = Quantity(np.logspace(3, 4, nbins), 'GeV')
    actual = effarea.evaluate(offset=offset, energy=energy).shape
    desired = np.zeros(nbins).shape
    assert_equal(actual, desired)

    # Case 3: offset = None, energy = scalar
    energy = Quantity(1.1, 'TeV')
    actual = effarea.evaluate(energy=energy).shape
    desired = effarea.offset.shape
    assert_equal(actual, desired)

    # Case 4: offset = 1Darray, energy = scalar
    energy = Quantity(1.5, 'TeV')
    nbins = 4
    offset = Angle(np.linspace(0, 1, nbins), 'degree')
    actual = effarea.evaluate(offset=offset, energy=energy).shape
    desired = np.zeros(nbins).shape
    assert_equal(actual, desired)

    # case 5: offset = 1Darray, energy = 1Darray
    nbinse = 50
    nbinso = 10
    offset = Angle(np.linspace(0, 1, nbinso), 'degree')
    energy = Quantity(np.logspace(0, 1, nbinse), 'TeV')
    actual = effarea.evaluate(offset=offset, energy=energy).shape
    desired = np.zeros([nbinso, nbinse]).shape
    assert_equal(actual, desired)

    # Test ARF export
    offset = Angle(0.236, 'degree')
    e_axis = Quantity(np.logspace(0, 1, 20), 'TeV')
    energy_lo = e_axis[:-1]
    energy_hi = e_axis[1:]

    effareafrom2d = effarea.to_effective_area_table(offset, energy_lo, energy_hi)

    energy = Quantity(np.sqrt(energy_lo.value * energy_hi.value), 'TeV')
    area = effarea.evaluate(offset, energy)
    effarea1d = EffectiveAreaTable(energy_lo, energy_hi, area)

    test_energy = Quantity(2.34, 'TeV')
    actual = effareafrom2d.effective_area_at_energy(test_energy)
    desired = effarea1d.effective_area_at_energy(test_energy)
    assert_equal(actual, desired)

    #Test ARF export #2
    effareafrom2dv2 = effarea.to_effective_area_table(offset)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.utils.data import get_pkg_data_filename
from astropy.tests.helper import pytest
from astropy.io import fits
from ...utils.testing import requires_dependency, requires_data
from ...utils.energy import EnergyBounds
from ...irf import EffectiveAreaTable2D, EffectiveAreaTable, abramowski_effective_area
from ...datasets import gammapy_extra


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


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_EffectiveAreaTable():
    filename = get_pkg_data_filename('data/arf_info.txt')
    info_str = open(filename, 'r').read()

    filename = gammapy_extra.filename('datasets/hess-crab4_pha/arf_run23526.fits')
    arf = EffectiveAreaTable.read(filename)
    assert arf.info() == info_str

    assert (arf.evaluate() == arf.effective_area).all() == True


@requires_data('gammapy-extra')
def test_EffectiveAreaTable_write(tmpdir):
    filename = gammapy_extra.filename('test_datasets/unbundled/irfs/arf.fits')
    irf = EffectiveAreaTable.read(filename)

    filename = str(tmpdir / 'effarea_test.fits')
    irf.write(filename)

    hdu_list = fits.open(filename)
    # TODO: replace this with an assert that tests more.
    assert len(hdu_list) == 2


INTERPOLATION_METHODS = ['linear']


@pytest.mark.parametrize(('method'), INTERPOLATION_METHODS)
@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_EffectiveAreaTable2D(method):
    filename = gammapy_extra.filename('datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_aeff_2d_023523.fits.gz')
    aeff = EffectiveAreaTable2D.read(filename)

    filename = get_pkg_data_filename('data/aeff2D_info.txt')
    info_str = open(filename, 'r').read()

    assert aeff.info() == info_str

    aeff.interpolation_method = method

    # Check that nodes are evaluated correctly
    e_node = 42
    off_node = 3
    offset = aeff.offset[off_node]
    energy = aeff.ebounds.log_centers[e_node]
    actual = aeff.evaluate(offset, energy)
    desired = aeff.eff_area[off_node, e_node]
    assert_allclose(actual, desired)

    # Check that values between node make sense
    energy2 = aeff.ebounds.log_centers[e_node + 1]
    upper = aeff.evaluate(offset, energy)
    lower = aeff.evaluate(offset, energy2)
    e_val = (energy + energy2) / 2
    actual = aeff.evaluate(offset, e_val)
    assert_equal(lower > actual and actual > upper, True)

    # Test evaluate function (return shape)
    # Case 0; offset = scalar, energy = scalar, done

    # Case 1: offset = scalar, energy = None
    offset = Angle(0.234, 'deg')
    actual = aeff.evaluate(offset=offset).shape
    desired = aeff.ebounds.log_centers.shape
    assert_equal(actual, desired)

    # Case 2: offset = scalar, energy = 1Darray
    offset = Angle(0.564, 'deg')
    nbins = 42
    energy = Quantity(np.logspace(3, 4, nbins), 'GeV')
    actual = aeff.evaluate(offset=offset, energy=energy).shape
    desired = np.zeros(nbins).shape
    assert_equal(actual, desired)

    # Case 3: offset = None, energy = scalar
    energy = Quantity(1.1, 'TeV')
    actual = aeff.evaluate(energy=energy).shape
    desired = aeff.offset.shape
    assert_equal(actual, desired)

    # Case 4: offset = 1Darray, energy = scalar
    energy = Quantity(1.5, 'TeV')
    nbins = 4
    offset = Angle(np.linspace(0, 1, nbins), 'deg')
    actual = aeff.evaluate(offset=offset, energy=energy).shape
    desired = np.zeros(nbins).shape
    assert_equal(actual, desired)

    # case 5: offset = 1Darray, energy = 1Darray
    nbinse = 50
    nbinso = 10
    offset = Angle(np.linspace(0, 1, nbinso), 'deg')
    energy = Quantity(np.logspace(0, 1, nbinse), 'TeV')
    actual = aeff.evaluate(offset=offset, energy=energy).shape
    desired = np.zeros([nbinso, nbinse]).shape
    assert_equal(actual, desired)

    # case 6: offset = 2Darray, energy = 1Darray
    nbinse = 16
    nx, ny = (12, 3)
    offset = np.linspace(1, 0, nx * ny).reshape(nx, ny)
    offset = Angle(offset, 'deg')
    energy = Quantity(np.logspace(0, 1, nbinse), 'TeV')
    actual = aeff.evaluate(offset=offset, energy=energy).shape
    desired = np.zeros([nx, ny, nbinse]).shape
    assert_equal(actual, desired)

    # Test ARF export
    offset = Angle(0.236, 'deg')
    e_axis = Quantity(np.logspace(0, 1, 20), 'TeV')

    effareafrom2d = aeff.to_effective_area_table(offset, e_axis)

    energy = EnergyBounds(e_axis).log_centers
    area = aeff.evaluate(offset, energy)
    effarea1d = EffectiveAreaTable(e_axis, area)

    test_energy = Quantity(2.34, 'TeV')
    actual = effareafrom2d.evaluate(test_energy)
    desired = effarea1d.evaluate(test_energy)
    assert_equal(actual, desired)

    # Test ARF export #2
    effareafrom2dv2 = aeff.to_effective_area_table(offset)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.tests.helper import pytest
from astropy.units import Quantity
from numpy.testing import assert_allclose, assert_equal

from ...datasets import gammapy_extra
from ...irf import EffectiveAreaTable, abramowski_effective_area
from ...utils.energy import EnergyBounds
from ...utils.testing import requires_dependency, requires_data, data_manager

def test_EffectiveArea2D():
    # These are just some quick and dirty tests they should be removed later
    from ...irf.effective_area import EffectiveArea2D
    energy = Quantity(np.logspace(0,1,4), 'TeV') 
    offset = Quantity([0.2, 0.3], 'deg') 
    effective_area = Quantity(np.arange(6).reshape(2,3))
    aeff = EffectiveArea2D(energy=energy, offset=offset, data=effective_area)
   
    #For now just test if subclass behaves correctly
    e = Quantity(1.1, 'TeV')
    o = Quantity(0.29, 'deg')
    idx = aeff.find_node(energy=e, offset=o)
    assert idx[0] == 1
    assert idx[1] == 0

    filename = '/home/kingj/Software/gammapy-extra/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_aeff_2d_023523.fits.gz' 
    aeff = EffectiveArea2D.read(filename)


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_EffectivateAreaTable2D(data_manager):
    # Check that nodes are evaluated correctly
    store = data_manager['hess-crab4-hd-hap-prod2']
    aeff = store.obs(obs_id=23523).aeff

    e_node = 43
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
    effareafrom2dv2 = aeff.to_effective_area_table('1.2 deg')
    actual = effareafrom2dv2.effective_area
    desired = aeff.evaluate(offset='1.2 deg')
    assert_equal(actual, desired)


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_EffectiveAreaTable(tmpdir, data_manager):

    store = data_manager['hess-crab4-hd-hap-prod2']
    aeff = store.obs(obs_id=23523).aeff
    arf = aeff.to_effective_area_table('0.3 deg')

    assert (arf.evaluate() == arf.effective_area).all() == True

    filename = gammapy_extra.filename('test_datasets/unbundled/irfs/arf.fits')
    irf = EffectiveAreaTable.read(filename)

    filename = str(tmpdir / 'effarea_test.fits')
    irf.write(filename)

    hdu_list = fits.open(filename)
    assert len(hdu_list) == 2


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

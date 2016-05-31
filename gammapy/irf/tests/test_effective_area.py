# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from numpy.testing import assert_allclose, assert_equal
from astropy.tests.helper import pytest
from ...datasets import gammapy_extra
from ...utils.testing import requires_dependency, requires_data, data_manager
from ...irf.effective_area import (
    EffectiveAreaTable2D, EffectiveAreaTable, abramowski_effective_area
)


@requires_dependency('scipy')
def test_EffectiveAreaTable2D_generic():
    # This tests NDData subclassing. Not needed for other IRF classes 

    # Exercise __init__  method 
    energy = np.logspace(0, 1, 4) * u.TeV
    offset = [0.2, 0.3] * u.deg
    effective_area = np.arange(6).reshape(3, 2) * u.cm * u.cm
    meta = dict(name = 'example')
    aeff = EffectiveAreaTable2D(offset=offset, energy=energy, data=effective_area,
                           meta=meta)
    assert (aeff.axes[0].data == energy).all()
    assert (aeff.axes[1].data == offset).all()
    assert (aeff.data == effective_area).all()
    assert aeff.meta.name == 'example'

    wrong_data = np.arange(8).reshape(4,2) * u.cm * u.cm

    with pytest.raises(ValueError):
        aeff.data = wrong_data
        aeff = EffectiveAreaTable(offset=offset, energy=energy, data=wrong_data)

    # Test evaluate function 
    # Check that nodes are evaluated correctly
    e_node = 1 
    off_node = 0
    offset = aeff.offset.nodes[off_node]
    energy = aeff.energy.nodes[e_node]
    actual = aeff.evaluate(offset=offset, energy=energy, method='nearest')
    desired = aeff.data[e_node, off_node]
    assert_allclose(actual, desired)

    actual = aeff.evaluate(offset=offset, energy=energy, method='linear')
    desired = aeff.data[e_node, off_node]
    assert_allclose(actual, desired)

    # Check that values between node make sense
    energy2 = aeff.energy.nodes[e_node + 1]
    upper = aeff.evaluate(offset=offset, energy=energy)
    lower = aeff.evaluate(offset=offset, energy=energy2)
    e_val = (energy + energy2) / 2
    actual = aeff.evaluate(offset=offset, energy=e_val)
    assert_equal(lower > actual and actual > upper, True)

    # Test return shape
    # Case 0; offset = scalar, energy = scalar, done

    # Case 1: offset = scalar, energy = None
    offset = 0.234 * u.deg
    actual = aeff.evaluate(offset=offset).shape
    desired = aeff.energy.nodes.shape
    assert_equal(actual, desired)

    # Case 2: offset = scalar, energy = 1Darray

    offset = 0.564 * u.deg
    nbins = 10 
    energy = np.logspace(3, 4, nbins) * u.GeV
    actual = aeff.evaluate(offset=offset, energy=energy).shape
    desired = np.zeros(nbins).shape
    assert_equal(actual, desired)

    # Case 3: offset = None, energy = scalar
    energy = 1.1 * u.TeV
    actual = aeff.evaluate(energy=energy).shape
    desired = tuple([aeff.offset.nbins])
    assert_equal(actual, desired)

    # Case 4: offset = 1Darray, energy = scalar
    energy = 1.5 * u.TeV
    nbins = 4
    offset = np.linspace(0, 1, nbins) * u.deg
    actual = aeff.evaluate(offset=offset, energy=energy).shape
    desired = np.zeros(nbins).shape
    assert_equal(actual, desired)

    # case 5: offset = 1Darray, energy = 1Darray
    nbinse = 5
    nbinso = 3
    offset = np.linspace(0.2, 0.3, nbinso) * u.deg
    energy = np.logspace(0, 1, nbinse) * u.TeV
    actual = aeff.evaluate(offset=offset, energy=energy).shape
    desired = np.zeros([nbinse, nbinso]).shape
    assert_equal(actual, desired)

    # case 6: offset = 2Darray, energy = 1Darray
    nbinse = 4
    nx, ny = (12, 3)
    offset = np.linspace(0.2, 0.3, nx * ny).reshape(nx, ny) * u.deg
    energy = np.logspace(0, 1, nbinse) * u.TeV
    actual = aeff.evaluate(offset=offset, energy=energy).shape
    desired = np.zeros([nbinse, nx, ny]).shape
    assert_equal(actual, desired)

    # Misc functions
    assert 'EffectiveAreaTable2D' in str(aeff)


@requires_dependency('scipy')
@requires_dependency('matplotlib')
@requires_data('gammapy-extra')
def test_EffectiveAreaTable2D(tmpdir):

    filename = gammapy_extra.filename('datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_aeff_2d_023523.fits.gz')
    aeff = EffectiveAreaTable2D.read(filename)

    assert aeff.energy.nbins == 73
    assert aeff.offset.nbins == 6
    assert aeff.data.shape == (73, 6)

    assert aeff.energy.unit == 'TeV'
    assert aeff.offset.unit == 'deg'
    assert aeff.data.unit == 'm2'

    assert_allclose(aeff.high_threshold.value, 99.083, atol=1e-2)
    assert_allclose(aeff.low_threshold.value, 0.603, atol=1e-2)
    
    test_e = 14 * u.TeV
    test_o = 0.2 * u.deg
    test_val = aeff.evaluate(energy=test_e, offset=test_o)
    assert_allclose(test_val.value, 740929.645, atol=1e-2)

    aeff.plot_image()
    aeff.plot_energy_dependence()
    aeff.plot_offset_dependence()

    # Test ARF export
    offset = 0.236  * u.deg
    e_axis = np.logspace(0, 1, 20) * u.TeV
    effareafrom2d = aeff.to_effective_area_table(offset, e_axis)

    energy = np.sqrt(e_axis[:-1] * e_axis[1:])
    area = aeff.evaluate(offset=offset, energy=energy)
    effarea1d = EffectiveAreaTable(energy=e_axis, data=area)

    test_energy = 2.34 * u.TeV
    actual = effareafrom2d.evaluate(energy=test_energy)
    desired = effarea1d.evaluate(energy=test_energy)
    assert_equal(actual, desired)

    # Test ARF export #2
    offset = 1.2 * u.deg
    effareafrom2dv2 = aeff.to_effective_area_table(offset=offset)
    actual = effareafrom2dv2.data
    desired = aeff.evaluate(offset=offset)
    assert_equal(actual, desired)


@requires_dependency('scipy')
@requires_dependency('matplotlib')
@requires_data('gammapy-extra')
def test_EffectiveAreaTable(tmpdir, data_manager):
    store = data_manager['hess-crab4-hd-hap-prod2']
    aeff = store.obs(obs_id=23523).aeff
    arf = aeff.to_effective_area_table(offset = 0.3 * u.deg)

    assert (arf.evaluate() == arf.data).all()
    
    arf.plot()

    filename = str(tmpdir / 'effarea_test.fits')
    arf.write(filename)

    arf2 = EffectiveAreaTable.read(filename)

    assert (arf.evaluate() == arf2.evaluate()).all()

def test_abramowski_effective_area():
    energy = 100 * u.GeV
    area_ref = 1.65469579e+07 * u.cm * u.cm 

    area = abramowski_effective_area(energy, 'HESS')
    assert_allclose(area, area_ref)
    assert area.unit == area_ref.unit

    energy = [0.1, 2] * u.TeV
    area_ref = [1.65469579e+07, 1.46451957e+09] * u.cm * u.cm

    area = abramowski_effective_area(energy, 'HESS')
    assert_allclose(area, area_ref)
    assert area.unit == area_ref.unit

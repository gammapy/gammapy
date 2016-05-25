# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from numpy.testing import assert_allclose, assert_equal
from astropy.tests.helper import pytest
from ...datasets import gammapy_extra
from ...utils.testing import requires_dependency, requires_data
from ...irf.effective_area import EffectiveArea2D


@requires_dependency('scipy')
def test_EffectiveArea2D_generic():
    # This tests NDData subclassing. Not needed for other IRF classes 

    # Exercise __init__  method 
    energy = np.logspace(0, 1, 4) * u.TeV
    offset = [0.2, 0.3] * u.deg
    effective_area = np.arange(6).reshape(3, 2) * u.cm * u.cm
    meta = dict(name = 'example')
    aeff = EffectiveArea2D(offset=offset, energy=energy, data=effective_area,
                           meta=meta)
    assert (aeff.axes[0].data == energy).all()
    assert (aeff.axes[1].data == offset).all()
    assert (aeff.data == effective_area).all()
    assert aeff.meta.name == 'example'

    wrong_data = np.arange(8).reshape(4,2) * u.cm * u.cm

    with pytest.raises(ValueError):
        aeff.data = wrong_data
        aeff = EffectiveArea(offset=offset, energy=energy, data=wrong_data)

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
    print(aeff)


@requires_dependency('scipy')
@requires_dependency('matplotlib')
@requires_data('gammapy-extra')
def test_EffectiveArea2D(tmpdir):

    filename = gammapy_extra.filename('datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_aeff_2d_023523.fits.gz')
    aeff = EffectiveArea2D.read(filename)

    assert aeff.energy.nbins == 73
    assert aeff.offset.nbins == 6
    assert aeff.data.shape == (73, 6)

    assert aeff.energy.unit == 'TeV'
    assert aeff.offset.unit == 'deg'
    assert aeff.data.unit == 'm2'

    aeff.plot_image()
    aeff.plot_energy_dependence()
    aeff.plot_offset_dependence()

    # Test ARF export
    # offset = Angle(0.236, 'deg')
    # e_axis = Quantity(np.logspace(0, 1, 20), 'TeV')
    # effareafrom2d = aeff.to_effective_area_table(offset, e_axis)
    # energy = EnergyBounds(e_axis).log_centers
    # area = aeff.evaluate(offset, energy)
    # effarea1d = EffectiveAreaTable(e_axis, area)
    # test_energy = Quantity(2.34, 'TeV')
    # actual = effareafrom2d.evaluate(test_energy)
    # desired = effarea1d.evaluate(test_energy)
    # assert_equal(actual, desired)

    # Test ARF export #2
    # effareafrom2dv2 = aeff.to_effective_area_table('1.2 deg')
    # actual = effareafrom2dv2.effective_area
    # desired = aeff.evaluate(offset='1.2 deg')
    # assert_equal(actual, desired)

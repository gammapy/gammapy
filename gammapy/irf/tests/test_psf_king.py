# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.tests.helper import pytest, assert_quantity_allclose
from astropy.units import Quantity
from astropy.coordinates import Angle
from ...utils.testing import requires_data
from ...irf import PSFKing


@pytest.fixture(scope='session')
def psf_king():
    filename = '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_psf_king_023523.fits.gz'
    return PSFKing.read(filename)


@requires_data('gammapy-extra')
def test_psf_king_evaluate(psf_king):
    energy = Quantity(1, "TeV")
    off1 = Angle(0, "deg")
    off2 = Angle(1, "deg")
    param_off1 = psf_king.evaluate(energy, off1)
    param_off2 = psf_king.evaluate(energy, off2)

    assert_quantity_allclose(param_off1["gamma"], psf_king.gamma[0, 8])
    assert_quantity_allclose(param_off2["gamma"], psf_king.gamma[2, 8])
    assert_quantity_allclose(param_off1["sigma"], psf_king.sigma[0, 8])
    assert_quantity_allclose(param_off2["sigma"], psf_king.sigma[2, 8])


@requires_data('gammapy-extra')
def test_psf_king_to_table(psf_king):
    theta1 = Angle(0, "deg")
    theta2 = Angle(1, "deg")
    psf_king_table_off1 = psf_king.to_energy_dependent_table_psf(theta=theta1)
    psf_king_table_off2 = psf_king.to_energy_dependent_table_psf(theta=theta2)
    offset = Angle(1, "deg")
    # energy = Quantity(1, "TeV") match with bin number 8
    # offset equal 1 degre match with the bin 200 in the psf_table
    value_off1 = psf_king.evaluate_direct(offset, psf_king.gamma[0, 8], psf_king.sigma[0, 8])
    value_off2 = psf_king.evaluate_direct(offset, psf_king.gamma[2, 8], psf_king.sigma[2, 8])
    # Test that the value at 1 degree in the histogram for the energy 1 Tev and theta=0 or 1 degree is equal to the one
    # obtained from the self.evaluate_direct() method at 1 degree
    assert_quantity_allclose(psf_king_table_off1.psf_value[8, 200], value_off1)
    assert_quantity_allclose(psf_king_table_off2.psf_value[8, 200], value_off2)

    # Test that the integral value is close to one
    bin_off = (psf_king_table_off1.offset[1] - psf_king_table_off1.offset[0])
    integral = np.sum(psf_king_table_off1.psf_value[8] * 2 * np.pi * psf_king_table_off1.offset * bin_off)
    assert_quantity_allclose(integral, 1, rtol=1e-1)


@requires_data('gammapy-extra')
def test_psf_king_write(psf_king, tmpdir):
    filename = str(tmpdir / 'king.fits')
    psf_king.write(filename)
    psf_king2 = PSFKing.read(filename)
    assert_quantity_allclose(psf_king2.energy, psf_king.energy)
    assert_quantity_allclose(psf_king2.offset, psf_king.offset)
    assert_quantity_allclose(psf_king2.gamma, psf_king.gamma)
    assert_quantity_allclose(psf_king2.sigma, psf_king.sigma)

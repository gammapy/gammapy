# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from astropy.coordinates import Angle
from gammapy.irf import PSFKing
from gammapy.utils.testing import assert_quantity_allclose, requires_data


@pytest.fixture(scope="session")
def psf_king():
    return PSFKing.read("$GAMMAPY_DATA/tests/hess_psf_king_023523.fits.gz")


@requires_data()
def test_psf_king_evaluate(psf_king):
    param_off1 = psf_king.evaluate(energy="1 TeV", offset="0 deg")
    param_off2 = psf_king.evaluate("1 TeV", "1 deg")

    assert_quantity_allclose(param_off1["gamma"], psf_king.gamma[0, 8])
    assert_quantity_allclose(param_off2["gamma"], psf_king.gamma[2, 8])
    assert_quantity_allclose(param_off1["sigma"], psf_king.sigma[0, 8])
    assert_quantity_allclose(param_off2["sigma"], psf_king.sigma[2, 8])


@requires_data()
def test_psf_king_to_table(psf_king):
    theta1 = Angle(0, "deg")
    theta2 = Angle(1, "deg")
    psf_king_table_off1 = psf_king.to_energy_dependent_table_psf(theta=theta1)
    psf_king_table_off2 = psf_king.to_energy_dependent_table_psf(theta=theta2)
    offset = Angle(1, "deg")
    # energy = Quantity(1, "TeV") match with bin number 8
    # offset equal 1 degre match with the bin 200 in the psf_table
    value_off1 = psf_king.evaluate_direct(
        offset, psf_king.gamma[0, 8], psf_king.sigma[0, 8]
    )
    value_off2 = psf_king.evaluate_direct(
        offset, psf_king.gamma[2, 8], psf_king.sigma[2, 8]
    )
    # Test that the value at 1 degree in the histogram for the energy 1 Tev and theta=0 or 1 degree is equal to the one
    # obtained from the self.evaluate_direct() method at 1 degree
    assert_quantity_allclose(psf_king_table_off1.psf_value[8, 200], value_off1)
    assert_quantity_allclose(psf_king_table_off2.psf_value[8, 200], value_off2)

    # Test that the integral value is close to one
    bin_off = psf_king_table_off1.rad_axis.bin_width[0]

    integral = np.sum(
        psf_king_table_off1.psf_value[8]
        * 2
        * np.pi
        * psf_king_table_off1.rad_axis.center
        * bin_off
    )
    assert_quantity_allclose(integral, 1, atol=0.03)


@requires_data()
def test_psf_king_write(psf_king, tmp_path):
    psf_king.write(tmp_path / "tmp.fits")
    psf_king2 = PSFKing.read(tmp_path / "tmp.fits")

    assert_quantity_allclose(
        psf_king2.energy_axis_true.edges, psf_king.energy_axis_true.edges
    )
    assert_quantity_allclose(psf_king2.offset_axis.center, psf_king.offset_axis.center)
    assert_quantity_allclose(psf_king2.gamma, psf_king.gamma)
    assert_quantity_allclose(psf_king2.sigma, psf_king.sigma)

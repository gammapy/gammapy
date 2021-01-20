# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from astropy.coordinates import Angle
from astropy import units as u
from gammapy.irf import PSFKing
from gammapy.utils.testing import assert_quantity_allclose, requires_data


@pytest.fixture(scope="session")
def psf_king():
    return PSFKing.read("$GAMMAPY_DATA/tests/hess_psf_king_023523.fits.gz")


@requires_data()
def test_psf_king_evaluate(psf_king):
    param_off1 = psf_king.evaluate_parameters(energy_true=1 * u.TeV, offset=0 * u.deg)
    param_off2 = psf_king.evaluate_parameters(energy_true=1 * u.TeV, offset=1 * u.deg)

    assert_allclose(param_off1["gamma"].value, 1.733179, rtol=1e-5)
    assert_allclose(param_off2["gamma"].value, 1.812795, rtol=1e-5)
    assert_allclose(param_off1["sigma"], 0.040576 * u.deg, rtol=1e-5)
    assert_allclose(param_off2["sigma"], 0.040765 * u.deg, rtol=1e-5)


@requires_data()
def test_psf_king_to_table(psf_king):
    theta1 = Angle(0, "deg")
    theta2 = Angle(1, "deg")
    psf_king_table_off1 = psf_king.to_energy_dependent_table_psf(offset=theta1)
    rad = Angle(1, "deg")
    # energy = Quantity(1, "TeV") match with bin number 8
    # offset equal 1 degre match with the bin 200 in the psf_table
    value_off1 = psf_king.evaluate(
        rad=rad, energy_true=1 * u.TeV, offset=theta1
    )
    value_off2 = psf_king.evaluate(
        rad=rad, energy_true=1 * u.TeV, offset=theta2
    )
    # Test that the value at 1 degree in the histogram for the energy 1 Tev and theta=0 or 1 degree is equal to the one
    # obtained from the self.evaluate_direct() method at 1 degree
    assert_allclose(0.005234 * u.Unit("deg-2"), value_off1, rtol=1e-4)
    assert_allclose(0.004015 * u.Unit("deg-2"), value_off2, rtol=1e-4)

    # Test that the integral value is close to one
    integral = psf_king_table_off1.containment(rad=1 *u.deg, energy_true=1 * u.TeV)
    assert_allclose(integral, 1, atol=3e-2)


@requires_data()
def test_psf_king_write(psf_king, tmp_path):
    psf_king.write(tmp_path / "tmp.fits")
    psf_king2 = PSFKing.read(tmp_path / "tmp.fits")

    assert_quantity_allclose(
        psf_king2.axes["energy_true"].edges, psf_king.axes["energy_true"].edges
    )
    assert_quantity_allclose(psf_king2.axes["offset"].center, psf_king.axes["offset"].center)
    assert_quantity_allclose(psf_king2.data["gamma"], psf_king.data["gamma"])
    assert_quantity_allclose(psf_king2.data["sigma"], psf_king.data["sigma"])

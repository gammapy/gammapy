# Licensed under a 3-clause BSD style license - see LICENSE.rst
from copy import deepcopy
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from gammapy.irf import EnergyDependentMultiGaussPSF, PSFKing
from gammapy.utils.testing import mpl_plot_check, requires_data


@requires_data()
class TestEnergyDependentMultiGaussPSF:
    @pytest.fixture(scope="session")
    def psf(self):
        filename = "$GAMMAPY_DATA/tests/unbundled/irfs/psf.fits"
        return EnergyDependentMultiGaussPSF.read(filename, hdu="POINT SPREAD FUNCTION")

    def test_info(self, psf):
        info_str = open(get_pkg_data_filename("./data/psf_info.txt")).read()

        print(psf.info())
        assert psf.info() == info_str

    def test_write(self, tmp_path, psf):
        psf.write(tmp_path / "tmp.fits")

        with fits.open(tmp_path / "tmp.fits", memmap=False) as hdu_list:
            assert len(hdu_list) == 2

    def test_to_table_psf(self, psf):
        energy = 1 * u.TeV
        theta = 0 * u.deg

        containment = [0.68, 0.8, 0.9]
        desired = psf.containment_radius(
            energy_true=energy, offset=theta, fraction=containment
        )

        assert_allclose(desired, [0.14775, 0.18675, 0.25075] * u.deg, rtol=1e-3)

    def test_to_unit(self, psf):
        with pytest.raises(NotImplementedError):
            psf.to_unit("deg-2")

    def test_to_psf3d(self, psf):
        rads = np.linspace(0.0, 1.0, 101) * u.deg
        psf_3d = psf.to_psf3d(rads)

        rad_axis = psf_3d.axes["rad"]
        assert rad_axis.nbin == 100
        assert rad_axis.unit == "deg"

        theta = 0.5 * u.deg
        energy = 0.5 * u.TeV

        containment = [0.68, 0.8, 0.9]
        desired = psf.containment_radius(
            energy_true=energy, offset=theta, fraction=containment
        )
        actual = psf_3d.containment_radius(
            energy_true=energy, offset=theta, fraction=containment
        )
        assert_allclose(np.squeeze(desired), actual, atol=0.005)

        # test default case
        psf_3d_def = psf.to_psf3d()
        assert psf_3d_def.axes["rad"].nbin == 66

    def test_eq(self, psf):
        psf1 = deepcopy(psf)
        assert psf1 == psf

        psf1.data[0][0] = 10
        assert not psf1 == psf

    def test_peek(self, psf):
        with mpl_plot_check():
            psf.peek()


@requires_data()
def test_psf_cta_1dc():
    filename = (
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    psf_irf = EnergyDependentMultiGaussPSF.read(filename, hdu="POINT SPREAD FUNCTION")

    # Check that PSF is filled with 0 for energy / offset where no PSF info is given.
    # This is needed so that stacked PSF computation doesn't error out,
    # trying to interpolate for observations / energies where this occurs.
    value = psf_irf.evaluate(
        energy_true=0.05 * u.TeV, rad=0.03 * u.deg, offset=4.5 * u.deg
    )
    assert_allclose(value, 0 * u.Unit("deg-2"))

    # Check that evaluation works for an energy / offset where an energy is available
    radius = psf_irf.containment_radius(
        fraction=0.68, energy_true=1 * u.TeV, offset=2 * u.deg
    )
    assert_allclose(radius, 0.052841 * u.deg, atol=1e-4)


@requires_data()
def test_get_sigmas_and_norms():
    filename = "$GAMMAPY_DATA/cta-caldb/Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits.gz"  # noqa: E501

    psf_irf = EnergyDependentMultiGaussPSF.read(filename, hdu="POINT SPREAD FUNCTION")

    value = psf_irf.evaluate(
        energy_true=1 * u.TeV, rad=0.03 * u.deg, offset=3.5 * u.deg
    )
    assert_allclose(value, 78.25826069 * u.Unit("deg-2"))


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
def test_psf_king_containment_radius(psf_king):
    radius = psf_king.containment_radius(
        fraction=0.68, energy_true=1 * u.TeV, offset=0.0 * u.deg
    )

    assert_allclose(radius, 0.14575 * u.deg, rtol=1e-5)


@requires_data()
def test_psf_king_evaluate_2(psf_king):
    theta1 = Angle(0, "deg")
    theta2 = Angle(1, "deg")
    rad = Angle(1, "deg")
    # energy = Quantity(1, "TeV") match with bin number 8
    # offset equal 1 degre match with the bin 200 in the psf_table
    value_off1 = psf_king.evaluate(rad=rad, energy_true=1 * u.TeV, offset=theta1)
    value_off2 = psf_king.evaluate(rad=rad, energy_true=1 * u.TeV, offset=theta2)
    # Test that the value at 1 degree in the histogram for the energy 1 Tev and
    # theta=0 or 1 degree is equal to the one obtained from the self.evaluate_direct()
    # method at 1 degree
    assert_allclose(0.005234 * u.Unit("deg-2"), value_off1, rtol=1e-4)
    assert_allclose(0.004015 * u.Unit("deg-2"), value_off2, rtol=1e-4)


@requires_data()
def test_psf_king_write(psf_king, tmp_path):
    psf_king.write(tmp_path / "tmp.fits")
    psf_king2 = PSFKing.read(tmp_path / "tmp.fits")

    assert_allclose(
        psf_king2.axes["energy_true"].edges, psf_king.axes["energy_true"].edges
    )
    assert_allclose(psf_king2.axes["offset"].center, psf_king.axes["offset"].center)
    assert_allclose(psf_king2.data["gamma"], psf_king.data["gamma"])
    assert_allclose(psf_king2.data["sigma"], psf_king.data["sigma"])

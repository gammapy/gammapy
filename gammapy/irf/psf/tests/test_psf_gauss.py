# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose   
from astropy import units as u
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from gammapy.irf import EnergyDependentMultiGaussPSF
from gammapy.utils.testing import mpl_plot_check, requires_data, requires_dependency


@requires_data()
class TestEnergyDependentMultiGaussPSF:
    @pytest.fixture(scope="session")
    def psf(self):
        filename = "$GAMMAPY_DATA/tests/unbundled/irfs/psf.fits"
        return EnergyDependentMultiGaussPSF.read(filename, hdu="POINT SPREAD FUNCTION")

    def test_info(self, psf):
        info_str = open(get_pkg_data_filename("data/psf_info.txt")).read()

        assert psf.info() == info_str

    def test_write(self, tmp_path, psf):
        psf.write(tmp_path / "tmp.fits")

        with fits.open(tmp_path / "tmp.fits", memmap=False) as hdu_list:
            assert len(hdu_list) == 2

    def test_to_table_psf(self, psf):
        energy = 1 * u.TeV
        theta = 0 * u.deg

        rad = np.linspace(0, 2, 300) * u.deg
        table_psf = psf.to_energy_dependent_table_psf(theta, rad=rad)

        psf_at_energy = psf.psf_at_energy_and_theta(energy, theta)

        containment = [0.68, 0.8, 0.9]
        desired = [psf_at_energy.containment_radius(_) for _ in containment]

        table_psf_at_energy = table_psf.table_psf_at_energy(energy)
        actual = table_psf_at_energy.containment_radius(containment)

        assert_allclose(desired, actual.degree, rtol=1e-2)

    def test_to_psf3d(self, psf):
        rads = np.linspace(0.0, 1.0, 101) * u.deg
        psf_3d = psf.to_psf3d(rads)
        assert psf_3d.rad_axis.nbin == 100
        assert psf_3d.rad_axis.unit == "deg"

        theta = 0.5 * u.deg
        energy = 0.5 * u.TeV

        containment = [0.68, 0.8, 0.9]
        desired = np.array(
            [psf.containment_radius(energy, theta, _).value for _ in containment]
        )
        actual = np.array(
            [psf_3d.containment_radius(energy, theta, _).value for _ in containment]
        )
        assert_allclose(np.squeeze(desired), actual, atol=0.005)

    @requires_dependency("matplotlib")
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
    psf = psf_irf.to_energy_dependent_table_psf("4.5 deg")
    psf = psf.table_psf_at_energy("0.05 TeV")
    assert_allclose(psf.evaluate(rad="0.03 deg").value, 0)

    # Check that evaluation works for an energy / offset where an energy is available
    psf = psf_irf.to_energy_dependent_table_psf("2 deg")
    psf = psf.table_psf_at_energy("1 TeV")
    assert_allclose(psf.containment_radius(0.68).deg, 0.052841, atol=1e-4)

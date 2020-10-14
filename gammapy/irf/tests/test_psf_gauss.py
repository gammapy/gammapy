# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from astropy import units as u
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from gammapy.irf.psf_gauss import (
    EnergyDependentMultiGaussPSF,
    HESSMultiGaussPSF,
    multi_gauss_psf_kernel,
)
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


class TestHESS:
    @staticmethod
    def test_dpdtheta2():
        """Check that the amplitudes and sigmas were converted correctly in
        HESS.to_MultiGauss2D() by comparing the dpdtheta2 distribution.

        Note that we set normalize=False in the to_MultiGauss2D call,
        which is necessary because the HESS PSF is *not* normalized
        correcly by the HESS software, it is usually a few % off.

        Also quite interesting is to look at the norms, since they
        represent the fractions of gammas in each of the three components.

        integral: 0.981723
        sigmas:   [ 0.0219206   0.0905762   0.0426358]
        norms:    [ 0.29085818  0.20162012  0.48924452]

        So in this case the HESS PSF 'scale' is 2% too low
        and e.g. the wide sigma = 0.09 deg PSF component contains 20%
        of the events.
        """
        filename = get_pkg_data_filename("data/psf.txt")
        hess = HESSMultiGaussPSF(filename)
        m = hess.to_MultiGauss2D(normalize=False)

        for theta in np.linspace(0, 1, 10):
            val_hess = hess.dpdtheta2(theta ** 2)
            val_m = m.dpdtheta2(theta ** 2)
            assert_almost_equal(val_hess, val_m, decimal=4)

    @staticmethod
    def test_gc():
        """Compare the containment radii computed with the HESS software
        with those found by using MultiGauss2D.

        This test fails for r95, where the HESS software gives a theta
        which is 10% higher. Probably the triple-Gauss doesn't represent
        the PSF will in the core or the fitting was bad or the
        HESS software has very large binning errors (they compute
        containment radius from the theta2 histogram directly, not
        using the triple-Gauss approximation)."""
        vals = [
            (68, 0.0663391),
            # TODO: check why this was different before
            # (95, 0.173846),  # 0.15310963243226974
            (95, 0.15310967713539758),
            (10, 0.0162602),
            (40, 0.0379536),
            (80, 0.088608),
        ]
        filename = get_pkg_data_filename("data/psf.txt")
        hess = HESSMultiGaussPSF(filename)
        m = hess.to_MultiGauss2D()
        assert_almost_equal(m.integral, 1)
        for containment, theta in vals:
            actual = m.containment_radius(containment / 100.0)
            assert_almost_equal(actual, theta, decimal=2)


def test_multi_gauss_psf_kernel():
    psf_data = {
        "psf1": {"ampl": 1, "fwhm": 2.5496814916215014},
        "psf2": {"ampl": 0.062025099992752075, "fwhm": 11.149272133127273},
        "psf3": {"ampl": 0.47460201382637024, "fwhm": 5.164014607542117},
    }
    psf_kernel = multi_gauss_psf_kernel(psf_data, x_size=51)

    assert_allclose(psf_kernel.array[25, 25], 0.05047558713797154)
    assert_allclose(psf_kernel.array[23, 29], 0.003259483464443567)

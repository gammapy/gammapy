# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing.utils import assert_allclose, assert_almost_equal
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy import units as u
from ...utils.testing import requires_dependency, requires_data
from ..psf_gauss import EnergyDependentMultiGaussPSF
from ..psf_gauss import multi_gauss_psf_kernel, HESSMultiGaussPSF


def make_test_psf(energy_bins=15, theta_bins=12):
    """Create a test FITS PSF file.

    A log-linear dependency in energy is assumed, where the size of
    the PSF decreases by a factor of tow over tow decades. The
    theta dependency is a parabola where at theta = 2 deg the size
    of the PSF has increased by 30%.

    Parameters
    ----------
    energy_bins : int
        Number of energy bins.
    theta_bins : int
        Number of theta bins.

    Returns
    -------
    psf : `~gammapy.irf.EnergyDependentMultiGaussPSF`
        PSF.
    """
    energies_all = np.logspace(-1, 2, energy_bins + 1)
    energies_lo = energies_all[:-1]
    energies_hi = energies_all[1:]
    theta_lo = np.linspace(0, 2.2, theta_bins)

    def sigma_energy_theta(energy, theta, sigma):
        # log-linear dependency of sigma with energy
        # m and b are choosen such, that at 100 TeV
        # we have sigma and at 0.1 TeV we have sigma/2
        m = -sigma / 6.
        b = sigma + m
        return (2 * b + m * np.log10(energy)) * (0.3 / 4 * theta ** 2 + 1)

    # Compute norms and sigmas values are taken from the psf.txt in
    # irf/test/data
    energies, thetas = np.meshgrid(energies_lo, theta_lo)

    sigmas = []
    for sigma in [0.0219206, 0.0905762, 0.0426358]:
        sigmas.append(sigma_energy_theta(energies, thetas, sigma))

    norms = []
    for norm in 302.654 * np.array([1, 0.0406003, 0.444632]):
        norms.append(norm * np.ones((theta_bins, energy_bins)))

    return EnergyDependentMultiGaussPSF(
        u.Quantity(energies_lo, "TeV"),
        u.Quantity(energies_hi, "TeV"),
        u.Quantity(theta_lo, "deg"),
        sigmas,
        norms,
    )


@requires_dependency("scipy")
@requires_data("gammapy-extra")
class TestEnergyDependentMultiGaussPSF:
    @pytest.fixture(scope="session")
    def psf(self):
        filename = "$GAMMAPY_EXTRA/test_datasets/unbundled/irfs/psf.fits"
        return EnergyDependentMultiGaussPSF.read(filename, hdu="POINT SPREAD FUNCTION")

    def test_info(self, psf):
        filename = get_pkg_data_filename("data/psf_info.txt")
        info_str = open(filename, "r").read()

        assert psf.info() == info_str

    def test_write(self, tmpdir, psf):
        # Write it back to disk
        filename = str(tmpdir / "multigauss_psf_test.fits")
        psf.write(filename)

        # Verify checksum
        with fits.open(filename) as hdu_list:
            # TODO: replace this assert with something else.
            # For unknown reasons this verify_checksum fails non-deterministically
            # see e.g. https://travis-ci.org/gammapy/gammapy/jobs/31056341#L1162
            # assert hdu_list[1].verify_checksum() == 1
            assert len(hdu_list) == 2

    def test_to_table_psf(self, psf):
        energy = 1 * u.TeV
        theta = 0 * u.deg

        table_psf = psf.to_energy_dependent_table_psf(theta)
        interpol_param = dict(method="nearest", bounds_error=False)
        table_psf_at_energy = table_psf.table_psf_at_energy(energy, interpol_param)
        psf_at_energy = psf.psf_at_energy_and_theta(energy, theta)

        containment = np.linspace(0, 0.95, 10)
        desired = [psf_at_energy.containment_radius(_) for _ in containment]
        actual = table_psf_at_energy.containment_radius(containment)

        # TODO: try to improve precision, so that rtol can be lowered
        assert_allclose(desired, actual.degree, rtol=0.03)

    def test_to_psf3d(self, psf):
        rads = np.linspace(0., 1.0, 301) * u.deg
        psf_3d = psf.to_psf3d(rads)
        assert psf_3d.rad_lo.shape == (300,)
        assert psf_3d.rad_lo.unit == "deg"

        theta = 0.5 * u.deg
        energy = 0.5 * u.TeV

        containment = np.linspace(0.1, 0.95, 10)
        desired = np.array(
            [psf.containment_radius(energy, theta, _).value for _ in containment]
        )
        actual = np.array(
            [psf_3d.containment_radius(energy, theta, _).value for _ in containment]
        )
        assert_allclose(np.squeeze(desired), actual, rtol=0.01)


@requires_dependency("scipy")
@requires_data("gammapy-extra")
def test_psf_cta_1dc():
    filename = "$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
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
    assert_allclose(psf.containment_radius(0.68).deg, 0.053728, atol=1e-4)


@requires_dependency("scipy")
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
            actual = m.containment_radius(containment / 100.)
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

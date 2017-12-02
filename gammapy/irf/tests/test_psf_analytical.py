# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing.utils import assert_allclose
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy import units as u
from ...utils.testing import requires_dependency, requires_data
from ...irf import EnergyDependentMultiGaussPSF


@requires_dependency('scipy')
@requires_data('gammapy-extra')
class TestEnergyDependentMultiGaussPSF:

    @pytest.fixture(scope='session')
    def psf(self):
        filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/irfs/psf.fits'
        return EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')

    def test_info(self, psf):
        filename = get_pkg_data_filename('data/psf_info.txt')
        info_str = open(filename, 'r').read()

        assert psf.info() == info_str

    def test_write(self, tmpdir, psf):
        # Write it back to disk
        filename = str(tmpdir / 'multigauss_psf_test.fits')
        psf.write(filename)

        # Verify checksum
        hdu_list = fits.open(filename)

        # TODO: replace this assert with something else.
        # For unknown reasons this verify_checksum fails non-deterministically
        # see e.g. https://travis-ci.org/gammapy/gammapy/jobs/31056341#L1162
        # assert hdu_list[1].verify_checksum() == 1
        assert len(hdu_list) == 2

    def test_to_table_psf(self, psf):
        energy = 1 * u.TeV
        theta = 0 * u.deg

        table_psf = psf.to_energy_dependent_table_psf(theta)
        interpol_param = dict(method='nearest', bounds_error=False)
        table_psf_at_energy = table_psf.table_psf_at_energy(energy, interpol_param)
        psf_at_energy = psf.psf_at_energy_and_theta(energy, theta)

        containment = np.linspace(0, 0.95, 10)
        desired = [psf_at_energy.containment_radius(_) for _ in containment]
        actual = table_psf_at_energy.containment_radius(containment)

        # TODO: try to improve precision, so that rtol can be lowered
        assert_allclose(desired, actual.degree, rtol=0.03)

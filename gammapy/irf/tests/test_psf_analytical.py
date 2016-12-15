# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing.utils import assert_allclose
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.units import Quantity
from astropy.tests.helper import pytest
from astropy.coordinates import Angle
from ...utils.testing import requires_dependency, requires_data
from ...irf import EnergyDependentMultiGaussPSF
from ...datasets import gammapy_extra

ENERGIES = Quantity([1, 10, 25], 'TeV')


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_EnergyDependentMultiGaussPSF():
    filename = get_pkg_data_filename('data/psf_info.txt')
    info_str = open(filename, 'r').read()

    filename = gammapy_extra.filename('test_datasets/unbundled/irfs/psf.fits')
    psf = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
    assert psf.info() == info_str


@requires_data('gammapy-extra')
def test_EnergyDependentMultiGaussPSF_write(tmpdir):
    filename = gammapy_extra.filename('test_datasets/unbundled/irfs/psf.fits')
    psf = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')

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


@requires_dependency('scipy')
@requires_data('gammapy-extra')
@pytest.mark.parametrize(('energy'), ENERGIES)
def test_to_table_psf(energy):
    filename = gammapy_extra.filename('datasets/hess-crab4-hd-hap-prod2/run023400-023599/'
                                      'run023523/hess_psf_3gauss_023523.fits.gz')
    psf = EnergyDependentMultiGaussPSF.read(filename, hdu='PSF_2D_GAUSS')
    theta = Angle(0, 'deg')

    table_psf = psf.to_energy_dependent_table_psf(theta)
    interpol_param = dict(method='nearest', bounds_error=False)
    table_psf_at_energy = table_psf.table_psf_at_energy(energy, interpol_param)
    psf_at_energy = psf.psf_at_energy_and_theta(energy, theta)

    containment = np.linspace(0, 0.95, 10)
    desired = [psf_at_energy.containment_radius(_) for _ in containment]
    actual = table_psf_at_energy.containment_radius(containment)

    # TODO: try to improve precision, so that rtol can be lowered
    assert_allclose(desired, actual.degree, rtol=0.03)

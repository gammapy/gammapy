# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from ...utils.testing import requires_dependency, requires_data
from ...irf import EnergyDependentMultiGaussPSF
from ...datasets import gammapy_extra


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

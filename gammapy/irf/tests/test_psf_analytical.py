# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.tests.helper import pytest
from astropy.utils.data import get_pkg_data_filename
from ...irf import EnergyDependentMultiGaussPSF

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_EnergyDependentMultiGaussPSF():
    from gammapy.datasets import psf_fits_table
    filename = get_pkg_data_filename('data/psf_info.txt')
    info_str = open(filename, 'r').read()
    psf = EnergyDependentMultiGaussPSF.from_fits(psf_fits_table())
    assert psf.info() == info_str


def test_EnergyDependentMultiGaussPSF_write():
    from gammapy.datasets import psf_fits_table
    from tempfile import NamedTemporaryFile
    from astropy.io import fits

    # Read test psf file
    psf = EnergyDependentMultiGaussPSF.from_fits(psf_fits_table())

    # Write it back to disk
    psf_file = NamedTemporaryFile(suffix='.fits').name
    psf.write(psf_file)

    # Verify checksum
    hdu_list = fits.open(psf_file)

    # TODO: replace this assert with something else.
    # For unknown reasons this verify_checksum fails non-deterministically
    # see e.g. https://travis-ci.org/gammapy/gammapy/jobs/31056341#L1162
    # assert hdu_list[1].verify_checksum() == 1
    assert len(hdu_list) == 2

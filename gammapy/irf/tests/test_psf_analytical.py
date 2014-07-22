# Licensed under a 3-clause BSD style license - see LICENSE.rst
from ..psf_analytical import EnergyDependentMultiGaussPSF

from astropy.tests.helper import pytest
from astropy.utils.data import get_pkg_data_filename

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
    assert hdu_list[1].verify_checksum() == 1
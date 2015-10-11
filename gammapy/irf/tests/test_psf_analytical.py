# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from ...irf import EnergyDependentMultiGaussPSF
from ...datasets import load_psf_fits_table

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_EnergyDependentMultiGaussPSF():
    filename = get_pkg_data_filename('data/psf_info.txt')
    info_str = open(filename, 'r').read()
    psf = EnergyDependentMultiGaussPSF.from_fits(load_psf_fits_table())
    assert psf.info() == info_str


def test_EnergyDependentMultiGaussPSF_write(tmpdir):
    # Read test psf file
    psf = EnergyDependentMultiGaussPSF.from_fits(load_psf_fits_table())

    # Write it back to disk
    filename = str(tmpdir.join('multigauss_psf_test.fits'))
    psf.write(filename)

    # Verify checksum
    hdu_list = fits.open(filename)

    # TODO: replace this assert with something else.
    # For unknown reasons this verify_checksum fails non-deterministically
    # see e.g. https://travis-ci.org/gammapy/gammapy/jobs/31056341#L1162
    # assert hdu_list[1].verify_checksum() == 1
    assert len(hdu_list) == 2

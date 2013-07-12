# Licensed under a 3-clause BSD style license - see LICENSE.rst
import subprocess
import pytest
from astropy.utils.data import get_pkg_data_filename
from ..sex import sex

try:
    subprocess.call('sex')
    HAS_SEX = True
except OSError:
    HAS_SEX = False


@pytest.mark.skipif('not HAS_SEX')
def test_sex():
    """Run SExtractor an example image and check number of detected sources"""
    filename = get_pkg_data_filename('../../data/counts.fits.gz')
    catalog, checkimage = sex(filename)
    assert len(catalog) == 42
    assert checkimage.data.max() == 42

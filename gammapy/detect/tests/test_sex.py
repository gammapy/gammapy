# Licensed under a 3-clause BSD style license - see LICENSE.rst
import subprocess
import tempfile
import pytest
from astropy.io import fits
from ...data import poisson_stats_image
from ..sex import sex

try:
    #subprocess.call('sex')
    process = subprocess.Popen('sex', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.communicate()
    HAS_SEX = True
except OSError:
    HAS_SEX = False


@pytest.mark.skipif('not HAS_SEX')
def test_sex():
    """Run SExtractor an example image and check number of detected sources"""
    # SExtractor can't process zipped files, so we can't simply use this:
    # from astropy.utils.data import get_pkg_data_filename
    # filename = get_pkg_data_filename('../../data/poisson_stats_image/counts.fits.gz')
    # Instead we make a non-zipped copy of the file:
    data = poisson_stats_image()
    filename = tempfile.mktemp('.fits')
    fits.writeto(filename, data=data)
    catalog, checkimage = sex(filename)
    assert len(catalog) == 35
    assert checkimage.data.max() == 35

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from ..psf_analytical import *

from astropy.utils.data import get_pkg_data_filename


def test_EnergyDependentMultiGaussPSF():
    filename = get_pkg_data_filename('data/psf_info.txt')
    info_str = open(filename, 'r').read()
    filename = get_pkg_data_filename('data/psf.fits')
    arf = EnergyDependentMultiGaussPSF.read(filename)
    assert arf.info() == info_str

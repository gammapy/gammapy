# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_allclose

from astropy.units import Quantity
from astropy.utils.data import get_pkg_data_filename
from astropy.tests.helper import pytest

from ..effective_area import EffectiveAreaTable, abramowski_effective_area
from ...datasets import arf_fits_table

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def test_abramowski_effective_area():
    energy = Quantity(100, 'GeV')
    area_ref = Quantity(1.65469579e+07, 'cm^2')

    area = abramowski_effective_area(energy, 'HESS')
    assert_allclose(area, area_ref)
    assert area.unit == area_ref.unit

    energy = Quantity([0.1, 2], 'TeV')
    area_ref = Quantity([1.65469579e+07, 1.46451957e+09], 'cm^2')

    area = abramowski_effective_area(energy, 'HESS')
    assert_allclose(area, area_ref)
    assert area.unit == area_ref.unit


@pytest.mark.skipif('not HAS_SCIPY')
def test_EffectiveAreaTable():
    filename = get_pkg_data_filename('data/arf_info.txt')
    info_str = open(filename, 'r').read()
    arf = EffectiveAreaTable.from_fits(arf_fits_table())
    assert arf.info() == info_str


def test_EffectiveAreaTable_write():
    from tempfile import NamedTemporaryFile
    from astropy.io import fits

    # Read test psf file
    psf = EffectiveAreaTable.from_fits(arf_fits_table())

    # Write it back to disk
    psf_file = NamedTemporaryFile(suffix='.fits').name
    psf.write(psf_file)

    # Verify checksum
    hdu_list = fits.open(psf_file)
    assert hdu_list[1].verify_checksum() == 1
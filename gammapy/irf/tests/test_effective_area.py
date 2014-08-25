# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from numpy.testing import assert_allclose
from astropy.units import Quantity
from astropy.utils.data import get_pkg_data_filename
from astropy.tests.helper import pytest
from ...irf import EffectiveAreaTable, abramowski_effective_area
from ...datasets import load_arf_fits_table

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
    arf = EffectiveAreaTable.from_fits(load_arf_fits_table())
    assert arf.info() == info_str


def test_EffectiveAreaTable_write():
    from tempfile import NamedTemporaryFile
    from astropy.io import fits

    # Read test psf file
    psf = EffectiveAreaTable.from_fits(load_arf_fits_table())

    # Write it back to disk
    with NamedTemporaryFile(suffix='.fits') as psf_file:
        psf.write(psf_file.name)

        # Verify checksum
        hdu_list = fits.open(psf_file.name)
        # TODO: replace this assert with something else.
        # For unknown reasons this verify_checksum fails non-deterministically
        # see e.g. https://travis-ci.org/gammapy/gammapy/jobs/31056341#L1162
        # assert hdu_list[1].verify_checksum() == 1
        assert len(hdu_list) == 2

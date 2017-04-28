# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest
from ...utils.testing import requires_data
from ... import datasets


@pytest.mark.xfail
@requires_data('gammapy-extra')
def test_load_catalog_atnf(tmpdir):
    catalog = datasets.load_catalog_atnf()
    assert len(catalog) == 2399

    # Check if catalog can be serialised to FITS
    filename = str(tmpdir / 'atnf_test.fits')
    catalog.write(filename)


@pytest.mark.xfail
@requires_data('gammapy-extra')
def test_load_catalog_green(tmpdir):
    catalog = datasets.load_catalog_green()
    assert len(catalog) == 294

    # Check if catalog can be serialised to FITS
    filename = str(tmpdir / 'green_test.fits')
    catalog.write(filename)

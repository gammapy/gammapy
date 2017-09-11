# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from ...utils.testing import requires_data
from ..snrcat import SourceCatalogSNRcat


@pytest.mark.xfail(reason="Fails locally")
@requires_data('gammapy-extra')
def test_load_catalog_snrcat(tmpdir):
    snrcat = SourceCatalogSNRcat()

    # Check SNR table
    table = snrcat.table
    assert len(table) > 300
    expected_colnames = ['Source_Name', 'RAJ2000']
    assert set(expected_colnames).issubset(table.colnames)
    # Check if catalog can be serialised to FITS
    filename = str(tmpdir / 'snrcat_test.fits')
    table.write(filename)

    # Check OBS table
    table = snrcat.obs_table
    assert len(table) > 1000
    expected_colnames = ['SNR_id', 'source_id']
    assert set(expected_colnames).issubset(table.colnames)

    # Check if catalog can be serialised to FITS
    filename = str(tmpdir / 'obs_test.fits')
    table.write(filename)

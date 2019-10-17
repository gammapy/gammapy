# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from gammapy.catalog import SourceCatalogSNRcat


@pytest.mark.skip(reason="see https://github.com/gammapy/gammapy/issues/2162")
@pytest.mark.remote_data
def test_load_catalog_snrcat(tmp_path):
    snrcat = SourceCatalogSNRcat()

    # Check SNR table
    table = snrcat.table
    assert len(table) > 300
    expected_colnames = ["Source_Name", "RAJ2000"]
    assert set(expected_colnames).issubset(table.colnames)
    # Check if catalog can be serialized to FITS
    table.write(tmp_path / "snrcat_test.fits")

    # Check OBS table
    table = snrcat.obs_table
    assert len(table) > 1000
    expected_colnames = ["SNR_id", "source_id"]
    assert set(expected_colnames).issubset(table.colnames)
    # Check if catalog can be serialized to FITS
    table.write(tmp_path / "obs_test.fits")

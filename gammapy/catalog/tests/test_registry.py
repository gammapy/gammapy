# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from ...utils.testing import requires_data
from ..registry import SourceCatalogRegistry
from ..gammacat import SourceCatalogGammaCat
from .test_core import make_test_catalog


@pytest.fixture()
def source_catalogs():
    cats = SourceCatalogRegistry.builtins()
    filename = "$GAMMAPY_DATA/catalogs/gammacat/gammacat.fits.gz"
    cats.register("gamma-cat", SourceCatalogGammaCat, (filename,))
    return cats


@requires_data()
def test_info_table(source_catalogs):
    table = source_catalogs.info_table
    assert table.colnames == ["name", "description", "sources"]


@requires_data()
def test_getitem(source_catalogs):
    cat = source_catalogs["2fhl"]
    assert cat.name == "2fhl"

    with pytest.raises(KeyError):
        source_catalogs["2FHL"]


def test_register(source_catalogs):
    source_catalogs.register(name="testcat", cls=make_test_catalog)

    assert "testcat" in source_catalogs.catalog_names
    cat = source_catalogs["testcat"]
    assert cat["bb"].name == "bb"


def test_source_catalogs(source_catalogs):
    """Test the global registry instance"""
    assert {"3fgl", "2fhl"}.issubset(source_catalogs.catalog_names)

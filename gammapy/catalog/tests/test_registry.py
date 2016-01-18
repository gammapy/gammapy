# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest
from ...utils.testing import requires_data
from ..registry import source_catalogs, SourceCatalogRegistry
from .test_core import make_test_catalog


class TestSourceCatalogs:
    """Test SourceCatalogRegistry directly (one instance per test),
    not the global `source_catalogs` registry.
    """

    def setup(self):
        self.source_catalogs = SourceCatalogRegistry.builtins()

    @requires_data('hgps')
    def test_info_table(self):
        table = self.source_catalogs.info_table
        assert table.colnames == ['Name', 'Description', 'Sources']

    @requires_data('gammapy-extra')
    def test_info(self):
        # TODO: assert output somehow
        self.source_catalogs.info()

    @requires_data('gammapy-extra')
    def test_getitem(self):
        cat = self.source_catalogs['2fhl']

        with pytest.raises(KeyError):
            source_catalogs['2FHL']

    def test_register(self):
        # catalog = make_test_catalog()
        self.source_catalogs.register(name='testcat', factory=make_test_catalog)

        assert 'testcat' in self.source_catalogs.catalog_names
        cat = self.source_catalogs['testcat']
        assert cat['bb'].name == 'bb'


def test_source_catalogs():
    """Test the global registry instance"""

    assert set(['3fgl', '2fhl']).issubset(source_catalogs.catalog_names)

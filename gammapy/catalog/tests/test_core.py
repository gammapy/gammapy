# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.table import Table
from ..core import SourceCatalog


def make_test_catalog():
    table = Table()
    table['Source_Name'] = ['a', 'bb', 'ccc']
    table['RA'] = [42.2, 43.3, 44.4]
    table['DEC'] = [1, 2, 3]

    catalog = SourceCatalog(table)

    return catalog


class TestSourceCatalog:
    def setup(self):
        self.cat = make_test_catalog()

    def test_table(self):
        assert_allclose(self.cat.table['RA'][1], 43.3)

    def test_row_index(self):
        idx = self.cat.row_index(name='bb')
        assert idx == 1

        with pytest.raises(KeyError):
            self.cat.row_index(name='invalid')

    def test_source_name(self):
        name = self.cat.source_name(index=1)
        assert name == 'bb'

        with pytest.raises(IndexError):
            self.cat.source_name(index=99)

        # This seems to raise IndexError or ValueError with
        # different Astropy versions, so we just check for
        # any exception here
        with pytest.raises(Exception):
            self.cat.source_name('invalid')

    def test_getitem(self):
        source = self.cat['a']
        assert source.data['Source_Name'] == 'a'

        source = self.cat[0]
        assert source.data['Source_Name'] == 'a'

        with pytest.raises(KeyError):
            self.cat['invalid']

        with pytest.raises(IndexError):
            self.cat[99]

        with pytest.raises(ValueError):
            self.cat[int]


class TestSourceCatalogObject:
    def setup(self):
        self.cat = make_test_catalog()
        self.source = self.cat['a']

    def test_name(self):
        assert self.source.name == 'a'

    def test_index(self):
        assert self.source.index == 0

    def test_pprint(self):
        # TODO: capture output and assert that it contains some substring
        self.source.pprint()

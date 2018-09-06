# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from collections import OrderedDict
import numpy as np
from numpy.testing import assert_allclose
from astropy.table import Table, Column
from astropy.units import Quantity
from ...utils.testing import assert_quantity_allclose
from ..core import SourceCatalog


def make_test_catalog():
    table = Table()
    table["Source_Name"] = ["a", "bb", "ccc"]
    table["RA"] = Column([42.2, 43.3, 44.4])
    table["DEC"] = Column([1, 2, 3], unit="deg")

    catalog = SourceCatalog(table)

    return catalog


class TestSourceCatalog:
    def setup(self):
        self.cat = make_test_catalog()

    def test_table(self):
        assert_allclose(self.cat.table["RA"][1], 43.3)

    def test_row_index(self):
        idx = self.cat.row_index(name="bb")
        assert idx == 1

        with pytest.raises(KeyError):
            self.cat.row_index(name="invalid")

    def test_source_name(self):
        name = self.cat.source_name(index=1)
        assert name == "bb"

        with pytest.raises(IndexError):
            self.cat.source_name(index=99)

        # This seems to raise IndexError or ValueError with
        # different Astropy versions, so we just check for
        # any exception here
        with pytest.raises(Exception):
            self.cat.source_name("invalid")

    def test_getitem(self):
        source = self.cat["a"]
        assert source.data["Source_Name"] == "a"

        source = self.cat[0]
        assert source.data["Source_Name"] == "a"

        source = self.cat[np.int(0)]
        assert source.data["Source_Name"] == "a"

        with pytest.raises(KeyError):
            self.cat["invalid"]

        with pytest.raises(IndexError):
            self.cat[99]

        with pytest.raises(ValueError):
            self.cat[int]

    def test_positions(self):
        positions = self.cat.positions
        assert len(positions) == 3


class TestSourceCatalogObject:
    def setup(self):
        self.cat = make_test_catalog()
        self.source = self.cat["bb"]

    def test_name(self):
        assert self.source.name == "bb"

    def test_index(self):
        assert self.source.index == 1

    def test_data(self):
        d = self.source.data
        assert isinstance(d, OrderedDict)
        assert isinstance(d["RA"], float)
        assert_allclose(d["RA"], 43.3)

        assert isinstance(d["DEC"], Quantity)
        assert_quantity_allclose(d["DEC"], Quantity(2, "deg"))

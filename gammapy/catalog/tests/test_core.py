# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.table import Column, Table
from astropy.units import Quantity
from gammapy.catalog import SourceCatalog
from gammapy.utils.testing import assert_quantity_allclose


class SomeSourceCatalog(SourceCatalog):
    """Minimal test source catalog class for unit tests."""

    name = "test123"
    tag = "test123"
    description = "Test source catalog"


def make_test_catalog():
    table = Table()
    table["Source_Name"] = ["a", "bb", "ccc"]
    table["RA"] = Column([42.2, 43.3, 44.4], unit="deg")
    table["DEC"] = Column([1, 2, 3], unit="deg")
    return SomeSourceCatalog(table)


class TestSourceCatalog:
    def setup(self):
        self.cat = make_test_catalog()

    def test_str(self):
        assert "description" in str(self.cat)
        assert "name" in str(self.cat)

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

        with pytest.raises(IndexError):
            self.cat.source_name("invalid")

    def test_getitem(self):
        source = self.cat["a"]
        assert source.data["Source_Name"] == "a"

        source = self.cat[0]
        assert source.data["Source_Name"] == "a"

        source = self.cat[np.int32(0)]
        assert source.data["Source_Name"] == "a"

        with pytest.raises(KeyError):
            self.cat["invalid"]

        with pytest.raises(IndexError):
            self.cat[99]

        with pytest.raises(TypeError):
            self.cat[1.2]

    def test_positions(self):
        positions = self.cat.positions
        assert len(positions) == 3

    def test_selection(self):
        new = self.cat[self.cat.table["Source_Name"] != "a"]
        assert len(new.table) == 2


class TestSourceCatalogObject:
    def setup(self):
        self.cat = make_test_catalog()
        self.source = self.cat["bb"]

    def test_name(self):
        assert self.source.name == "bb"

    def test_row_index(self):
        assert self.source.row_index == 1

    def test_data(self):
        d = self.source.data
        assert isinstance(d, dict)

        assert isinstance(d["RA"], Quantity)
        assert_quantity_allclose(d["RA"], Quantity(43.3, "deg"))

        assert isinstance(d["DEC"], Quantity)
        assert_quantity_allclose(d["DEC"], Quantity(2, "deg"))

    def test_position(self):
        position = self.source.position
        assert_allclose(position.ra.deg, 43.3)
        assert_allclose(position.dec.deg, 2)

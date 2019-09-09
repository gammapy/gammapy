# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.table import Column, Table
from gammapy.utils.table import (
    table_from_row_data,
    table_row_to_dict,
    table_standardise_units_copy,
)


def test_table_standardise_units():
    table = Table(
        [
            Column([1], "a", unit="ph cm-2 s-1"),
            Column([1], "b", unit="ct cm-2 s-1"),
            Column([1], "c", unit="cm-2 s-1"),
            Column([1], "d"),
        ]
    )

    table = table_standardise_units_copy(table)

    assert table["a"].unit == "cm-2 s-1"
    assert table["b"].unit == "cm-2 s-1"
    assert table["c"].unit == "cm-2 s-1"
    assert table["d"].unit is None


@pytest.fixture()
def table():
    return Table(
        [Column([1, 2], "a"), Column([1, 2] * u.m, "b"), Column(["x", "yy"], "c")]
    )


def test_table_row_to_dict(table):
    actual = table_row_to_dict(table[1])
    expected = {"a": 2, "b": 2 * u.m, "c": "yy"}
    assert actual == expected


def test_table_from_row_data():
    rows = [dict(a=1, b=1 * u.m, c="x"), dict(a=2, b=2 * u.km, c="yy")]
    table = table_from_row_data(rows)
    assert isinstance(table, Table)
    assert table["b"].unit == "m"
    assert_allclose(table["b"].data, [1, 2000])

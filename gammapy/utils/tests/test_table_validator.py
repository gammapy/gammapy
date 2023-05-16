# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Column, Table
from astropy.units import UnitTypeError
from gammapy.utils.table_validator import ColumnValidator, TableValidator


@pytest.fixture
def simple_table():
    nrows, nvals = 10, 5
    data = {
        "COL1": np.zeros(nrows, dtype=str),
        "COL2": np.zeros((nrows, nvals)) * u.s,
        "COL3": np.zeros(nrows) * u.TeV,
    }
    return Table(data)


def test_column_validator():
    column = Column(
        name="test",
        dtype="float",
        unit="m",
        shape=(10,),
        description="Some test column",
    )

    validator = ColumnValidator(dtype="float", unit="km", shape=(10,))
    res = validator.validate_column(column)

    assert res.name == column.name

    validator = ColumnValidator(dtype="float", unit="s", shape=(10,))
    with pytest.raises(UnitTypeError):
        validator.validate_column(column)


@pytest.mark.xfail
def test_column_validator_complex_types():
    column = Column(name="test", data=SkyCoord(0, 0, unit="deg", frame="icrs"))

    validator = ColumnValidator(dtype="SkyCoord", unit="deg")
    res = validator.validate_column(column)

    assert res.data.ra == 0


def test_column_from_definition():
    definition = ColumnValidator(dtype="float32", unit="m", description="test")

    column = definition.to_column("column")

    assert column.name == "column"
    assert column.unit == "m"
    assert column.dtype == "float32"
    assert column.description == "test"
    assert len(column) == 0


def test_validate_table(simple_table):
    validator = TableValidator(
        COL1=ColumnValidator(dtype="str", unit="", required=True),
        COL2=ColumnValidator(dtype="float", unit="s", shape=(5,), required=True),
        COL3=ColumnValidator(dtype="float", unit="GeV", required=True),
        COL4=ColumnValidator(dtype="float", unit="GeV"),
    )

    res = validator.run(simple_table)
    assert len(res) == 10


def test_validate_table_fail(simple_table):
    validator = TableValidator(
        COL3=ColumnValidator(dtype="str", unit=""),
        COL2=ColumnValidator(dtype="float", unit="s", shape="(5,)"),
        COL1=ColumnValidator(dtype="float64", unit="GeV"),
    )
    with pytest.raises(TypeError):
        validator.run(simple_table)


def test_table_from_definition():
    validator = TableValidator.from_builtin()

    table = validator.to_table()

    assert len(table.columns) == 5
    assert "EVENT_ID" in table.colnames
    assert "ENERGY" in table.colnames

    table = validator.to_table(include_optional=["MULTIP", "COREX"])

    assert len(table.columns) == 7
    assert "MULTIP" in table.colnames
    assert "COREX" in table.colnames

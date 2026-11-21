# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings
import numpy as np
import pytest
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from gammapy.io.fits.validation import (
    ColumnDescription,
    DefinitionValidator,
    columns_from_header,
    columns_from_table,
)

# ---------------------------------------------------------------------------
# DefinitionValidator: header keywords
# ---------------------------------------------------------------------------

KEYWORD_DEFINITION = {
    "A": {"dtype": "str", "required": True},
    "B": {"dtype": "int"},
    "C": {"dtype": ["int", "str"], "allowed": [1, "x"]},
    "MODE": {"dtype": "str"},
    "IF_DRIFT": {"dtype": "float", "required_if": {"MODE": "DRIFT"}},
    "UNLESS_DRIFT": {"dtype": "float", "required_unless": {"MODE": "DRIFT"}},
    "D": {"dtype": "str", "default": "dflt"},
}


def test_check_header_compliant():
    header = {"A": "a", "B": 2, "C": "x", "MODE": "POINTING", "UNLESS_DRIFT": 1.0}
    assert DefinitionValidator(KEYWORD_DEFINITION).check_header(header) == []


def test_check_header_missing_required():
    errors = DefinitionValidator(KEYWORD_DEFINITION).check_header({"UNLESS_DRIFT": 1.0})
    assert errors == ["Missing mandatory keyword(s): ['A']"]


@pytest.mark.parametrize("empty", ["", None, []])
def test_check_header_empty_value_is_absent(empty):
    errors = DefinitionValidator(KEYWORD_DEFINITION).check_header(
        {"A": empty, "UNLESS_DRIFT": 1.0}
    )
    assert errors == ["Missing mandatory keyword(s): ['A']"]


def test_check_header_conditional():
    validator = DefinitionValidator(KEYWORD_DEFINITION)

    errors = validator.check_header({"A": "a", "MODE": "DRIFT"})
    assert errors == ["['IF_DRIFT'] required when MODE='DRIFT'"]

    errors = validator.check_header({"A": "a", "MODE": "POINTING"})
    assert errors == ["['UNLESS_DRIFT'] required when MODE='POINTING'"]

    # absent condition keyword: the "unless" rule applies
    errors = validator.check_header({"A": "a"})
    assert errors == ["['UNLESS_DRIFT'] required when MODE=None"]

    assert validator.check_header({"A": "a", "MODE": "DRIFT", "IF_DRIFT": 1.0}) == []


def test_check_header_dtype():
    validator = DefinitionValidator(KEYWORD_DEFINITION)
    header = {"A": 1, "B": 2.5, "UNLESS_DRIFT": 1.0}
    errors = validator.check_header(header)
    assert "Invalid A=1: expected str" in errors
    assert "Invalid B=2.5: expected int" in errors

    # numpy scalars are accepted, booleans are not numbers
    assert (
        validator.check_header({"A": "a", "B": np.int16(2), "UNLESS_DRIFT": 1.0}) == []
    )
    assert validator.check_header({"A": "a", "B": True, "UNLESS_DRIFT": 1.0}) == [
        "Invalid B=True: expected int"
    ]
    # int accepted for float
    assert validator.check_header({"A": "a", "UNLESS_DRIFT": 1}) == []


def test_check_header_allowed():
    errors = DefinitionValidator(KEYWORD_DEFINITION).check_header(
        {"A": "a", "C": 2, "UNLESS_DRIFT": 1.0}
    )
    assert errors == ["Invalid C=2: allowed values [1, 'x']"]


def test_check_header_unknown_keywords_allowed():
    header = {"A": "a", "UNLESS_DRIFT": 1.0, "NOT_IN_DEFINITION": object()}
    assert DefinitionValidator(KEYWORD_DEFINITION).check_header(header) == []


def test_with_defaults():
    validator = DefinitionValidator(KEYWORD_DEFINITION)
    assert validator.with_defaults({"A": "a"}) == {"D": "dflt", "A": "a"}
    assert validator.with_defaults({"D": "mine"}) == {"D": "mine"}


# ---------------------------------------------------------------------------
# DefinitionValidator: table columns
# ---------------------------------------------------------------------------

COLUMN_DEFINITION = {
    "X": {"dtype": "float", "required": True, "unit": "TeV", "ndim": 1},
    "Y": {"dtype": "float", "required": True, "unit": "m2", "ndim": 2},
    "OPT": {"dtype": "float", "unit": "deg"},
}


def make_xy_table(x_unit="TeV", y_unit="m2", y_shape=(1, 2, 3)):
    table = Table()
    table["X"] = np.ones((1, 3)) * u.Unit(x_unit)
    table["Y"] = np.ones(y_shape) * u.Unit(y_unit)
    return table


def test_check_table_compliant():
    assert DefinitionValidator(COLUMN_DEFINITION).check_table(make_xy_table()) == []


def test_check_table_missing_required():
    table = make_xy_table()
    table.remove_column("Y")
    errors = DefinitionValidator(COLUMN_DEFINITION).check_table(table)
    assert errors == ["Missing mandatory column(s): ['Y']"]


def test_check_table_missing_columns_grouped():
    """All missing required columns are reported in one message, like keywords."""
    errors = DefinitionValidator(COLUMN_DEFINITION).check_table(Table())
    assert errors == ["Missing mandatory column(s): ['X', 'Y']"]


def test_check_table_ndim():
    table = make_xy_table(y_shape=(1, 3))
    errors = DefinitionValidator(COLUMN_DEFINITION).check_table(table)
    assert errors == ["Y: Column ndim incorrect. Expected 2, got 1."]


def test_check_table_unit():
    validator = DefinitionValidator(COLUMN_DEFINITION)
    errors = validator.check_table(make_xy_table(x_unit="kg"))
    assert errors == ["X: Column unit incorrect. Expected TeV, got kg."]

    # equivalent units are accepted
    assert validator.check_table(make_xy_table(x_unit="MeV", y_unit="cm2")) == []


def test_check_table_collects_all_errors():
    table = make_xy_table(x_unit="kg", y_unit="s", y_shape=(1, 3))
    errors = DefinitionValidator(COLUMN_DEFINITION).check_table(table)
    assert len(errors) == 3  # X unit, Y ndim and Y unit


def test_check_table_optional_column():
    validator = DefinitionValidator(COLUMN_DEFINITION)
    table = make_xy_table()
    table["OPT"] = [1.0] * u.kg
    assert validator.check_table(table) == [
        "OPT: Column unit incorrect. Expected deg, got kg."
    ]


def test_check_table_dtype_kind():
    definition = {"N": {"dtype": "float", "required": True}}
    validator = DefinitionValidator(definition)
    for dtype in (np.float32, np.float64):
        assert validator.check_table(Table({"N": np.ones(2, dtype=dtype)})) == []
    assert validator.check_table(Table({"N": np.ones(2, dtype=np.int32)})) == [
        "N: Column dtype incorrect. Expected float, got int32."
    ]


def test_check_table_dtype_exact():
    validator = DefinitionValidator({"N": {"dtype": "int64", "required": True}})
    assert validator.check_table(Table({"N": np.ones(2, dtype=np.int64)})) == []
    # byte order does not matter (FITS data are big-endian)
    assert validator.check_table(Table({"N": np.ones(2, dtype=">i8")})) == []
    assert validator.check_table(Table({"N": np.ones(2, dtype=np.int32)})) == [
        "N: Column dtype incorrect. Expected int64, got int32."
    ]


@pytest.mark.parametrize(
    "spec, values, valid",
    [
        ("int", np.ones(2, dtype=np.uint16), True),
        ("int", np.ones(2, dtype=bool), False),
        ("bool", np.ones(2, dtype=bool), True),
        ("str", np.array(["a", "b"]), True),
        ("str", np.ones(2), False),
        (["float", "int"], np.ones(2, dtype=np.int16), True),
        (["float", "int"], np.array(["a", "b"]), False),
    ],
)
def test_check_table_dtype_names(spec, values, valid):
    validator = DefinitionValidator({"N": {"dtype": spec}})
    assert (validator.check_table(Table({"N": values})) == []) is valid


def test_check_table_dtype_list_message():
    validator = DefinitionValidator({"N": {"dtype": ["float", "int"]}})
    assert validator.check_table(Table({"N": ["a", "b"]})) == [
        "N: Column dtype incorrect. Expected ['float', 'int'], got str."
    ]


def test_check_columns_bit_field():
    validator = DefinitionValidator({"FLAGS": {"dtype": "bit"}})
    bits = ColumnDescription(1, None, np.dtype(bool), bit_field=True)
    logical = ColumnDescription(1, None, np.dtype(bool), bit_field=False)
    assert validator.check_columns({"FLAGS": bits}) == []
    assert validator.check_columns({"FLAGS": logical}) == [
        "FLAGS: Column dtype incorrect. Expected bit, got bool."
    ]
    # in-memory table: the TFORM is unknown, any boolean column is accepted
    assert validator.check_table(Table({"FLAGS": np.zeros((2, 32), dtype=bool)})) == []
    # a bit field is not a "bool" column
    validator = DefinitionValidator({"FLAGS": {"dtype": "bool"}})
    assert validator.check_columns({"FLAGS": bits}) == [
        "FLAGS: Column dtype incorrect. Expected bool, got bit."
    ]


def test_check_table_dtype_not_given():
    """Columns whose spec has no dtype accept any type."""
    validator = DefinitionValidator({"N": {"required": True}})
    assert validator.check_table(Table({"N": np.array(["a", "b"])})) == []


def test_check_table_invalid_dtype_in_definition():
    validator = DefinitionValidator({"N": {"dtype": "not_a_dtype"}})
    errors = validator.check_table(Table({"N": [1.0]}))
    assert errors[0].startswith("N: invalid dtype 'not_a_dtype' in definition")


def test_check_table_no_unit_means_dimensionless():
    definition = {"N": {"required": True}}
    table = Table({"N": [1.0]})
    assert DefinitionValidator(definition).check_table(table) == []
    table["N"].unit = "deg"
    assert DefinitionValidator(definition).check_table(table) == [
        "N: Column unit incorrect. Expected , got deg."
    ]


def test_check_columns_conditional():
    definition = {"ALT": {"required_if": {"OBS_MODE": "DRIFT"}}}
    validator = DefinitionValidator(definition)
    columns = {}
    assert validator.check_columns(columns, {"OBS_MODE": "POINTING"}) == []
    assert validator.check_columns(columns, {"OBS_MODE": "DRIFT"}) == [
        "['ALT'] required when OBS_MODE='DRIFT'"
    ]


# ---------------------------------------------------------------------------
# Column descriptions from a FITS header (no data read)
# ---------------------------------------------------------------------------

FITS_COLUMNS = [
    fits.Column("SCALAR", "E", array=np.ones(2), unit="TeV"),
    fits.Column("REPEAT1", "1E", array=np.ones(2)),
    fits.Column("VECTOR", "5E", array=np.ones((2, 5)), unit="deg"),
    fits.Column("MATRIX", "6E", dim="(3,2)", array=np.ones((2, 2, 3)), unit="m2"),
    fits.Column("MATRIX1", "3E", dim="(3,1)", array=np.ones((2, 1, 3))),
    fits.Column("CUBE", "24E", dim="(4,3,2)", array=np.ones((2, 2, 3, 4))),
    fits.Column("STRING", "10A", array=np.array(["a", "b"])),
    fits.Column("STRINGS", "30A", dim="(10,3)", array=np.array([["a"] * 3] * 2)),
    fits.Column("BIT", "1X", array=np.array([[1], [0]], dtype=bool)),
    fits.Column("LOGICAL", "3L", array=np.ones((2, 3), dtype=bool)),
    fits.Column("VLA", "PE()", array=np.array([np.ones(3), np.ones(2)], dtype=object)),
    fits.Column("FITSUNIT", "E", array=np.ones(2), unit="s-1.MeV-1.sr-1"),
    fits.Column("INT64", "K", array=np.ones(2, dtype=np.int64)),
    fits.Column("INT16", "I", array=np.ones(2, dtype=np.int16)),
    fits.Column("UINT8", "B", array=np.ones(2, dtype=np.uint8)),
    fits.Column("DOUBLE", "D", array=np.ones(2)),
    fits.Column("BITS", "32X", array=np.zeros((2, 32), dtype=bool)),
    fits.Column("UINT64", "K", bzero=2**63, array=np.ones(2, dtype=np.uint64)),
    fits.Column("UINT32", "J", bzero=2**31, array=np.ones(2, dtype=np.uint32)),
    fits.Column("SCALED", "J", array=np.ones(2, dtype=np.int32)),  # TSCAL set below
    fits.Column("BADUNIT", "E", array=np.ones(2), unit="not_a_unit"),
]


@pytest.fixture(scope="module")
def fits_table_file(tmp_path_factory):
    filename = tmp_path_factory.mktemp("columns") / "columns.fits"
    fits.BinTableHDU.from_columns(FITS_COLUMNS, name="TEST").writeto(filename)
    # scaled integers are read as float64
    idx = [c.name for c in FITS_COLUMNS].index("SCALED") + 1
    fits.setval(filename, f"TSCAL{idx}", value=0.5, ext=1)
    return filename


@pytest.mark.parametrize("name", [c.name for c in FITS_COLUMNS])
def test_columns_from_header_matches_table_read(fits_table_file, name):
    header = fits.getheader(fits_table_file, "TEST")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", u.UnitsWarning)  # BADUNIT
        table = Table.read(fits_table_file, hdu="TEST")
    from_header = columns_from_header(header)[name]
    from_table = columns_from_table(table)[name]
    assert from_header.ndim == from_table.ndim
    assert str(from_header.unit) == str(from_table.unit)
    if from_table.dtype.kind in "SU":
        assert from_header.dtype.kind in "SU"
    else:
        assert from_header.dtype == from_table.dtype.newbyteorder("=")


def test_columns_from_header_signed_byte():
    """FITS standard: TFORM B with TZERO=-128 holds signed bytes (astropy: float64)."""
    header = fits.Header(
        {"TFIELDS": 1, "TTYPE1": "SBYTE", "TFORM1": "B", "TZERO1": -128}
    )
    assert columns_from_header(header)["SBYTE"].dtype == np.dtype("int8")


def test_columns_from_header_bit_field(fits_table_file):
    columns = columns_from_header(fits.getheader(fits_table_file, "TEST"))
    assert columns["BITS"].bit_field
    assert columns["BIT"].bit_field
    assert not columns["LOGICAL"].bit_field
    assert not columns["DOUBLE"].bit_field
    # in-memory tables do not know the TFORM
    table = Table.read(fits_table_file, hdu="TEST", unit_parse_strict="silent")
    assert columns_from_table(table)["BITS"].bit_field is None


def test_columns_from_header_no_columns():
    assert columns_from_header(fits.Header()) == {}


def test_column_description():
    column = ColumnDescription(ndim=2, unit=u.m**2)
    errors = DefinitionValidator(COLUMN_DEFINITION).check_columns(
        {"X": ColumnDescription(1, u.TeV), "Y": column}
    )
    # no dtype known: the dtype is not checked
    assert errors == []
    errors = DefinitionValidator(COLUMN_DEFINITION).check_columns(
        {"X": ColumnDescription(1, u.TeV, np.dtype("int32")), "Y": column}
    )
    assert errors == ["X: Column dtype incorrect. Expected float, got int32."]

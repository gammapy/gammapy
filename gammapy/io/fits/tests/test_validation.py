# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import warnings
import numpy as np
import pytest
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from gammapy.io.fits.validation import (
    ColumnDescription,
    DefinitionValidator,
    FormatComplianceError,
    FormatValidator,
    ValidationReport,
    columns_from_header,
    columns_from_table,
    format_class_keys,
    header_definition,
    validate_hdus,
)
from gammapy.utils.testing import requires_data

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


# ---------------------------------------------------------------------------
# Helpers: minimal GADF v0.3 compliant HDUs
# ---------------------------------------------------------------------------

EVENTS_META = {
    "HDUCLASS": "GADF",
    "HDUDOC": "https://gamma-astro-data-formats.readthedocs.io/en/v0.3/index.html",
    "HDUVERS": "0.3",
    "HDUCLAS1": "EVENTS",
    "OBS_ID": 1,
    "TSTART": 0.0,
    "TSTOP": 10.0,
    "ONTIME": 10.0,
    "LIVETIME": 9.0,
    "DEADC": 0.9,
    "EQUINOX": 2000.0,
    "RADECSYS": "ICRS",
    "ORIGIN": "Gammapy",
    "TELESCOP": "TEST",
    "INSTRUME": "TEST",
    "CREATOR": "Gammapy",
    "OBS_MODE": "POINTING",
    "RA_PNT": 83.6,
    "DEC_PNT": 22.0,
    "MJDREFI": 51910,
    "MJDREFF": 0.00074287037,
    "TIMEUNIT": "s",
    "TIMESYS": "TT",
    "TIMEREF": "LOCAL",
    "GEOLON": 16.5,
    "GEOLAT": -23.27,
    "ALTITUDE": 1835.0,
}

AEFF_META = {
    "HDUCLASS": "GADF",
    "HDUDOC": "https://gamma-astro-data-formats.readthedocs.io/en/v0.3/index.html",
    "HDUVERS": "0.3",
    "HDUCLAS1": "RESPONSE",
    "HDUCLAS2": "EFF_AREA",
    "HDUCLAS3": "FULL-ENCLOSURE",
    "HDUCLAS4": "AEFF_2D",
}


def make_events_table(meta=None):
    table = Table()
    table["EVENT_ID"] = np.arange(3, dtype=np.int64)
    table["TIME"] = [1.0, 2.0, 3.0] * u.s
    table["RA"] = [83.0, 83.5, 84.0] * u.deg
    table["DEC"] = [21.5, 22.0, 22.5] * u.deg
    table["ENERGY"] = [1.0, 2.0, 3.0] * u.TeV
    table.meta.update(EVENTS_META if meta is None else meta)
    return table


def make_aeff_table(meta=None, area_unit="m2"):
    table = Table()
    for name, unit in [
        ("ENERG_LO", "TeV"),
        ("ENERG_HI", "TeV"),
        ("THETA_LO", "deg"),
        ("THETA_HI", "deg"),
    ]:
        table[name] = np.ones((1, 4)) * u.Unit(unit)
    table["EFFAREA"] = np.ones((1, 2, 4)) * u.Unit(area_unit)
    table.meta.update(AEFF_META if meta is None else meta)
    return table


def events_meta(drop=(), **changes):
    meta = {k: v for k, v in EVENTS_META.items() if k not in drop}
    meta.update(changes)
    return meta


# ---------------------------------------------------------------------------
# FormatValidator
# ---------------------------------------------------------------------------


def test_format_validator_unknown_format():
    with pytest.raises(ValueError, match="not registered"):
        FormatValidator("NOT_A_FORMAT")


def test_validate_events_compliant():
    report = FormatValidator("GADF", "0.3").validate(make_events_table())
    assert isinstance(report, ValidationReport)
    assert report.hdu == "EVENTS"
    assert report.valid
    assert report.header_valid and report.table_valid
    assert report.errors == []


def test_validate_aeff_compliant():
    report = FormatValidator("GADF", "0.3").validate(make_aeff_table())
    assert report.hdu == "AEFF_2D"
    assert report.valid


def test_validate_hdu_class_given():
    # an IRF table without HDUCLASn keywords: class given explicitly
    report = FormatValidator("GADF", "0.3", log_errors=False).validate(
        make_aeff_table(meta={}), hdu="aeff_2d"
    )
    assert report.hdu == "AEFF_2D"
    assert report.table_valid
    assert not report.header_valid


def test_validate_meta_header_only():
    report = FormatValidator("GADF", "0.3").validate_meta(EVENTS_META, hdu="EVENTS")
    assert report.table_errors is None
    assert report.table_valid is None
    assert report.valid


def test_validate_table_errors():
    table = make_aeff_table(area_unit="kg")
    report = FormatValidator("GADF", "0.3", log_errors=False).validate(table)
    assert report.header_valid
    assert not report.table_valid
    assert report.errors == [
        "table: EFFAREA: Column unit incorrect. Expected m2, got kg."
    ]


def test_validate_missing_table_definition():
    report = FormatValidator("GADF", "0.3", log_errors=False).validate(
        Table({"A": [1]}), hdu="NOT_A_CLASS"
    )
    assert report.table_errors == ["No GADF v0.3 table definition for 'NOT_A_CLASS'"]


def test_events_time_and_location_required():
    meta = events_meta(drop=("MJDREFI", "TIMESYS", "GEOLON"))
    report = FormatValidator("GADF", "0.3", log_errors=False).validate_meta(
        meta, hdu="EVENTS"
    )
    assert report.header_errors == [
        "Missing mandatory keyword(s): ['MJDREFI', 'TIMESYS', 'GEOLON']"
    ]


def test_events_obs_mode_values():
    meta = events_meta(OBS_MODE="WOBBLE")
    v03 = FormatValidator("GADF", "0.3", log_errors=False)
    v02 = FormatValidator("GADF", "0.2", log_errors=False)
    assert v03.validate_meta(meta, hdu="EVENTS").header_errors == [
        "Invalid OBS_MODE='WOBBLE': allowed values "
        "['POINTING', 'RASTER', 'SLEW', 'SCAN', 'DRIFT']"
    ]
    assert v02.validate_meta(
        events_meta(HDUVERS="0.2", OBS_MODE="WOBBLE"), hdu="EVENTS"
    ).valid


def test_events_pointing_v02_v03():
    """GADF v0.3 allows drift scans (ALT/AZ_PNT), v0.2 always needs RA/DEC_PNT."""
    drift = events_meta(
        drop=("RA_PNT", "DEC_PNT"), OBS_MODE="DRIFT", ALT_PNT=70.0, AZ_PNT=0.0
    )
    v02 = FormatValidator("GADF", "0.2", log_errors=False)
    v03 = FormatValidator("GADF", "0.3", log_errors=False)

    assert v03.validate_meta(drift, hdu="EVENTS").valid
    assert v02.validate_meta(drift, hdu="EVENTS").header_errors == [
        "Missing mandatory keyword(s): ['RA_PNT', 'DEC_PNT']"
    ]

    drift_no_altaz = {k: v for k, v in drift.items() if k not in ("ALT_PNT", "AZ_PNT")}
    assert v03.validate_meta(drift_no_altaz, hdu="EVENTS").header_errors == [
        "['ALT_PNT', 'AZ_PNT'] required when OBS_MODE='DRIFT'"
    ]

    pointing_no_radec = events_meta(drop=("RA_PNT", "DEC_PNT"))
    assert v03.validate_meta(pointing_no_radec, hdu="EVENTS").header_errors == [
        "['RA_PNT', 'DEC_PNT'] required when OBS_MODE='POINTING'"
    ]


def test_version_from_hduvers():
    validator = FormatValidator("GADF", version=None)
    assert validator.validate_meta(events_meta(), hdu="EVENTS").version == "0.3"
    report = validator.validate_meta(events_meta(HDUVERS="0.2"), hdu="EVENTS")
    assert report.version == "0.2"


def test_unknown_hduvers():
    report = FormatValidator("GADF", "0.3", log_errors=False).validate_meta(
        events_meta(HDUVERS="9.9"), hdu="EVENTS"
    )
    assert report.header_errors == [
        "Invalid HDUVERS='9.9': allowed values ['0.2', '0.3']"
    ]


def test_unknown_class_key():
    """A class key with no definition is an error for the header and the table."""
    validator = FormatValidator("GADF", "0.3", log_errors=False)
    report = validator.validate(make_events_table(), hdu="NOT_A_CLASS")
    assert report.header_errors == ["No GADF v0.3 header definition for 'NOT_A_CLASS'"]
    assert report.table_errors == ["No GADF v0.3 table definition for 'NOT_A_CLASS'"]

    # unresolved from the header
    report = validator.validate_meta({"HDUCLASS": "GADF", "HDUCLAS1": "FOO"})
    assert report.hdu == "UNKNOWN_CLAS"
    assert not report.valid


def test_validate_events_column_dtypes(tmp_path):
    """GADF column types: EVENT_ID int64, TIME float64, RA "float", EVENT_TYPE bits."""
    table = make_events_table()
    table["EVENT_ID"] = table["EVENT_ID"].astype(np.int32)
    table["TIME"] = table["TIME"].astype(np.float32)
    table["RA"] = table["RA"].astype(np.float32)  # any float width
    hdu = fits.table_to_hdu(table)
    columns = fits.ColDefs(
        [fits.Column("EVENT_TYPE", "32L", array=np.zeros((3, 32), dtype=bool))]
    )
    hdu = fits.BinTableHDU.from_columns(hdu.columns + columns, header=hdu.header)

    report = FormatValidator("GADF", "0.3", log_errors=False).validate_hdu(hdu)
    assert report.table_errors == [
        "EVENT_ID: Column dtype incorrect. Expected int64, got int32.",
        "TIME: Column dtype incorrect. Expected float64, got float32.",
        "EVENT_TYPE: Column dtype incorrect. Expected bit, got bool.",
    ]


def test_validate_events_bit_field(tmp_path):
    hdu = fits.table_to_hdu(make_events_table())
    columns = fits.ColDefs(
        [fits.Column("EVENT_TYPE", "32X", array=np.zeros((3, 32), dtype=bool))]
    )
    hdu = fits.BinTableHDU.from_columns(hdu.columns + columns, header=hdu.header)
    report = FormatValidator("GADF", "0.3").validate_hdu(hdu)
    assert report.valid


def test_irf_columns_have_no_dtype():
    """GADF does not specify IRF column types: any float width (or int) passes."""
    table = make_aeff_table()
    table["EFFAREA"] = table["EFFAREA"].astype(np.float32)
    assert FormatValidator("GADF", "0.3").validate(table).valid


def test_strict_raises_with_full_report():
    meta = events_meta(drop=("OBS_MODE", "DEC_PNT"))
    with pytest.raises(FormatComplianceError) as excinfo:
        FormatValidator("GADF", "0.3", strict=True).validate_meta(meta, hdu="EVENTS")
    report = excinfo.value.report
    assert len(report.header_errors) == 2  # all errors collected before raising
    assert isinstance(excinfo.value, ValueError)


def test_errors_logged_unless_disabled(caplog):
    meta = events_meta(drop=("OBS_ID",))
    with caplog.at_level(logging.WARNING, logger="gammapy.io.fits.validation"):
        FormatValidator("GADF", "0.3").validate_meta(meta, hdu="EVENTS")
    assert "Missing mandatory keyword(s): ['OBS_ID']" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="gammapy.io.fits.validation"):
        FormatValidator("GADF", "0.3", log_errors=False).validate_meta(
            meta, hdu="EVENTS"
        )
    assert caplog.text == ""


def test_errors_logged_to_file(tmp_path):
    """Errors go through the standard logging module: a FileHandler saves them."""
    logfile = tmp_path / "validation.log"
    handler = logging.FileHandler(logfile)
    logger = logging.getLogger("gammapy.io.fits.validation")
    logger.addHandler(handler)
    try:
        FormatValidator("GADF", "0.3").validate_meta(
            events_meta(drop=("OBS_ID",)), hdu="EVENTS"
        )
    finally:
        logger.removeHandler(handler)
        handler.close()
    assert logfile.read_text() == (
        "EVENTS: header: Missing mandatory keyword(s): ['OBS_ID']\n"
    )


def test_report_str():
    report = ValidationReport(
        hdu="EVENTS",
        format="GADF",
        version="0.3",
        header_errors=["e1"],
        table_errors=[],
    )
    assert str(report) == "EVENTS (GADF v0.3): FAIL\n  - header: e1"


# ---------------------------------------------------------------------------
# Validation of FITS HDUs from their header
# ---------------------------------------------------------------------------


def test_validate_hdu_does_not_read_data(tmp_path):
    filename = tmp_path / "events.fits"
    make_events_table().write(filename)
    with fits.open(filename) as hdulist:
        report = FormatValidator("GADF", "0.3").validate_hdu(hdulist[1])
        assert "data" not in hdulist[1].__dict__
    assert report.valid
    assert report.table_valid  # columns checked from TTYPE / TFORM / TUNIT


def test_validate_hdu_same_as_table(tmp_path):
    filename = tmp_path / "aeff.fits"
    make_aeff_table(area_unit="kg").write(filename)
    validator = FormatValidator("GADF", "0.3", log_errors=False)
    with fits.open(filename) as hdulist:
        from_hdu = validator.validate_hdu(hdulist[1])
    from_table = validator.validate(Table.read(filename))
    assert from_hdu.header_errors == from_table.header_errors
    assert from_hdu.table_errors == from_table.table_errors


def test_validate_header():
    hdu = fits.table_to_hdu(make_events_table())
    report = FormatValidator("GADF", "0.3").validate_header(hdu.header)
    assert report.hdu == "EVENTS"
    assert report.table_valid


def test_validate_header_without_columns():
    header = fits.Header(EVENTS_META)
    report = FormatValidator("GADF", "0.3").validate_header(header, hdu="EVENTS")
    assert report.table_errors is None


def test_validate_image_hdu():
    hdu = fits.ImageHDU(np.zeros(12))
    hdu.header.update(HDUCLASS="GADF", HDUCLAS1="SKYMAP", PIXTYPE="HEALPIX")
    report = FormatValidator("GADF", "0.3", log_errors=False).validate_hdu(
        hdu, hdu="SKYMAP"
    )
    assert report.table_errors is None
    assert "Missing mandatory keyword(s): ['PIXTYPE']" not in report.header_errors


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------


def test_validate_hdus(tmp_path):
    good = tmp_path / "good.fits"
    hdulist = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.table_to_hdu(make_events_table()),
            fits.table_to_hdu(make_aeff_table()),
        ]
    )
    hdulist[1].name, hdulist[2].name = "EVENTS", "EFFECTIVE AREA"
    hdulist.writeto(good)

    entries = [
        (good, "EVENTS", "EVENTS"),
        (good, "EFFECTIVE AREA", "AEFF_2D"),
        (good, "EFFECTIVE AREA", None),  # class resolved from the header
        (good, "NOT_AN_HDU", "GTI"),
        (tmp_path / "missing.fits", "EVENTS", "EVENTS"),
    ]
    reports = validate_hdus(entries, format="GADF", version="0.3")

    assert [r.valid for r in reports] == [True, True, True, False, False]
    assert reports[2].hdu == "AEFF_2D"
    assert "cannot read HDU 'NOT_AN_HDU'" in reports[3].errors[0]
    assert "cannot open" in reports[4].errors[0]


def test_validate_hdus_mislabelled(tmp_path):
    filename = tmp_path / "events.fits"
    make_events_table().write(filename)
    (report,) = validate_hdus([(filename, 1, "GTI")], format="GADF", version="0.3")
    assert report.hdu == "GTI"
    assert report.table_errors == ["Missing mandatory column(s): ['START', 'STOP']"]


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def test_header_definition():
    definition = header_definition("GADF", "0.3", "EVENTS")
    assert definition["OBS_MODE"]["required"]
    assert header_definition("GADF", "0.3", "NOT_A_CLASS") is None
    assert header_definition("NOT_A_FORMAT", "0.3", "EVENTS") is None


def test_format_class_keys():
    keys = format_class_keys("GADF")
    assert {"EVENTS", "GTI", "POINTING", "AEFF_2D", "BKG_3D", "PSF_TABLE"} <= keys
    assert "THETA" in keys


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------


@requires_data()
def test_validate_hess_dl3_dr1_events():
    filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz"
    from gammapy.utils.scripts import make_path

    with fits.open(make_path(filename)) as hdulist:
        v02 = FormatValidator("GADF", "0.2").validate_hdu(hdulist["EVENTS"])
        v03 = FormatValidator("GADF", "0.3", log_errors=False).validate_hdu(
            hdulist["EVENTS"]
        )
    assert v02.valid
    assert v03.header_errors == [
        "Invalid OBS_MODE='WOBBLE': allowed values "
        "['POINTING', 'RASTER', 'SLEW', 'SCAN', 'DRIFT']"
    ]

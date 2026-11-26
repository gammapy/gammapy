# Licensed under a 3-clause BSD style license - see LICENSE.rst
import subprocess
import sys
import pytest
from gammapy.io.fits.registries import DATA_FORMATS_MODELS, _kw, compose_header
from gammapy.io.fits.registries.common import FITS_TIME_KEYWORDS
from gammapy.io.fits.registries.gadf import (
    GADF_EXTNAME_TO_TAG,
    GADF_HDUDOC,
    GADF_IRF_DL3_HDU_SPECIFICATION,
    GADF_PRODUCTS_HEADER_DEFINITION,
    GADF_PRODUCTS_TABLE_DEFINITION,
    gadf_hdu_class_key,
)

VERSIONS = ["0.2", "0.3"]


# ---------------------------------------------------------------------------
# compose_header
# ---------------------------------------------------------------------------


def test_compose_header():
    a = {"X": _kw("str"), "Y": _kw("int")}
    b = {"Y": _kw("float")}
    out = compose_header(a, b, required=("X", "Z"), updates={"Y": {"default": 1.0}})

    assert out == {
        "X": {"dtype": "str", "required": True},
        "Y": {"dtype": "float", "default": 1.0},
        "Z": {"required": True},
    }
    # blocks are copied, not modified
    assert a == {"X": {"dtype": "str"}, "Y": {"dtype": "int"}}


def test_fits_blocks_are_optional():
    assert not any(spec.get("required") for spec in FITS_TIME_KEYWORDS.values())


# ---------------------------------------------------------------------------
# GADF registration
# ---------------------------------------------------------------------------


def test_gadf_registered():
    models = DATA_FORMATS_MODELS["GADF"]
    assert {"TABLE", "HEADER", "HDU_CLASS_KEY"} <= set(models)
    assert models["TABLE"] is GADF_PRODUCTS_TABLE_DEFINITION
    assert models["HEADER"] is GADF_PRODUCTS_HEADER_DEFINITION
    assert models["HDU_CLASS_KEY"] is gadf_hdu_class_key


def test_registries_import_no_analysis_code():
    """The registries are pure data: importing them must not load gammapy.data/irf/maps."""
    code = (
        "import sys; import gammapy.io.fits.registries, gammapy.io.fits.validation; "
        "print([m for m in ('gammapy.data', 'gammapy.irf', 'gammapy.maps') "
        "if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


@pytest.mark.parametrize("version", VERSIONS)
def test_versions_complete(version):
    assert version in GADF_HDUDOC
    headers = GADF_PRODUCTS_HEADER_DEFINITION[version]
    tables = GADF_PRODUCTS_TABLE_DEFINITION[version]
    # every DL3 IRF class has a header and a table definition
    for spec in GADF_IRF_DL3_HDU_SPECIFICATION.values():
        key = spec["mandatory_keywords"]["HDUCLAS4"].upper()
        assert key in headers
        if key != "GTPSF":
            assert key in tables


@pytest.mark.parametrize("version", VERSIONS)
def test_header_defaults(version):
    events = GADF_PRODUCTS_HEADER_DEFINITION[version]["EVENTS"]
    assert events["HDUCLASS"]["default"] == "GADF"
    assert events["HDUVERS"]["default"] == version
    assert events["HDUDOC"]["default"] == GADF_HDUDOC[version]
    assert events["HDUCLAS1"]["default"] == "EVENTS"


def required(version, key):
    definition = GADF_PRODUCTS_HEADER_DEFINITION[version][key]
    return {kw for kw, spec in definition.items() if spec.get("required")}


@pytest.mark.parametrize("version", VERSIONS)
def test_events_required(version):
    expected = {
        "HDUCLASS", "HDUDOC", "HDUVERS", "HDUCLAS1",
        "OBS_ID", "TSTART", "TSTOP", "ONTIME", "LIVETIME", "DEADC",
        "EQUINOX", "RADECSYS", "ORIGIN", "TELESCOP", "INSTRUME", "CREATOR",
        "MJDREFI", "MJDREFF", "TIMEUNIT", "TIMESYS", "TIMEREF",
        "GEOLON", "GEOLAT", "ALTITUDE",
    }  # fmt: skip
    if version == "0.2":
        expected |= {"RA_PNT", "DEC_PNT"}
    else:
        expected |= {"OBS_MODE"}
    assert required(version, "EVENTS") == expected


def test_events_v03_pointing_rules():
    events = GADF_PRODUCTS_HEADER_DEFINITION["0.3"]["EVENTS"]
    for kw in ("RA_PNT", "DEC_PNT"):
        assert events[kw]["required_unless"] == {"OBS_MODE": "DRIFT"}
    for kw in ("ALT_PNT", "AZ_PNT"):
        assert events[kw]["required_if"] == {"OBS_MODE": "DRIFT"}
    assert events["OBS_MODE"]["allowed"] == [
        "POINTING", "RASTER", "SLEW", "SCAN", "DRIFT"
    ]  # fmt: skip
    assert "allowed" not in GADF_PRODUCTS_HEADER_DEFINITION["0.2"]["EVENTS"]["OBS_MODE"]


@pytest.mark.parametrize("version", VERSIONS)
def test_gti_pointing_required(version):
    time = {"MJDREFI", "MJDREFF", "TIMEUNIT", "TIMESYS", "TIMEREF"}
    location = {"GEOLON", "GEOLAT", "ALTITUDE"}
    assert time <= required(version, "GTI")
    assert not location & required(version, "GTI")
    assert time | location <= required(version, "POINTING")


@pytest.mark.parametrize("version", VERSIONS)
def test_irf_required(version):
    irf = {
        "HDUCLASS",
        "HDUDOC",
        "HDUVERS",
        "HDUCLAS1",
        "HDUCLAS2",
        "HDUCLAS3",
        "HDUCLAS4",
    }
    assert required(version, "AEFF_2D") == irf
    assert required(version, "BKG_3D") == irf | {"FOVALIGN"}
    assert required(version, "BKG_2D") == irf | {"FOVALIGN"}


def test_column_dtypes_from_gadf():
    """dtype only where GADF gives a column type (events, GTI, pointing, WCS bands)."""
    typed = {"EVENTS", "GTI", "POINTING", "IMAGE"}
    for version, tables in GADF_PRODUCTS_TABLE_DEFINITION.items():
        for key, columns in tables.items():
            has_dtype = {name for name, spec in columns.items() if "dtype" in spec}
            if key in typed:
                assert has_dtype == set(columns), (version, key)
            else:
                assert not has_dtype, (version, key)

    events = GADF_PRODUCTS_TABLE_DEFINITION["0.3"]["EVENTS"]
    assert events["EVENT_ID"]["dtype"] == "int64"
    assert events["TIME"]["dtype"] == "float64"
    assert events["RA"]["dtype"] == "float"
    assert events["EVENT_TYPE"]["dtype"] == "bit"
    for version in VERSIONS:
        tables = GADF_PRODUCTS_TABLE_DEFINITION[version]
        assert tables["GTI"]["START"]["dtype"] == "float64"
        assert tables["POINTING"]["TIME"]["dtype"] == "float64"


def test_gammaness_v03_only():
    assert "GAMMANESS" not in GADF_PRODUCTS_TABLE_DEFINITION["0.2"]["EVENTS"]
    assert "GAMMANESS" in GADF_PRODUCTS_TABLE_DEFINITION["0.3"]["EVENTS"]


@pytest.mark.parametrize("version", VERSIONS)
def test_every_table_class_has_header(version):
    """No table class key is left without a header definition."""
    tables = GADF_PRODUCTS_TABLE_DEFINITION[version]
    headers = GADF_PRODUCTS_HEADER_DEFINITION[version]
    assert set(tables) <= set(headers)
    assert "BASE" not in headers


def test_table_v03_derived_from_v02():
    v02 = GADF_PRODUCTS_TABLE_DEFINITION["0.2"]
    v03 = GADF_PRODUCTS_TABLE_DEFINITION["0.3"]
    assert v02["PSF_3GAUSS"]["SCALE"]["unit"] == ""
    assert v03["PSF_3GAUSS"]["SCALE"]["unit"] == "sr-1"
    assert v02["PSF_3GAUSS"] is not v03["PSF_3GAUSS"]


def test_definitions_use_known_spec_entries():
    known = {
        "dtype", "required", "required_if", "required_unless", "allowed",
        "default", "unit", "comment", "ndim", "description",
    }  # fmt: skip
    for registry in (GADF_PRODUCTS_HEADER_DEFINITION, GADF_PRODUCTS_TABLE_DEFINITION):
        for definitions in registry.values():
            for definition in definitions.values():
                for spec in definition.values():
                    assert set(spec) <= known


# ---------------------------------------------------------------------------
# HDU class key resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "meta, expected",
    [
        ({"HDUCLAS1": "EVENTS"}, "EVENTS"),
        ({"HDUCLAS1": "GTI"}, "GTI"),
        ({"HDUCLAS1": "RESPONSE", "HDUCLAS4": "AEFF_2D"}, "AEFF_2D"),
        ({"HDUCLAS4": "BKG_3D"}, "BKG_3D"),
        ({"HDUCLAS4": "PSF_TABLE"}, "PSF_TABLE"),
        ({"EXTNAME": "EVENTS"}, "EVENTS"),
        # HDUCLAS4 missing: guessed from a non-ambiguous EXTNAME
        ({"EXTNAME": "EFFECTIVE AREA"}, "AEFF_2D"),
        ({"EXTNAME": "ENERGY DISPERSION"}, "EDISP_2D"),
        ({"EXTNAME": "PSF_2D_TABLE"}, "PSF_TABLE"),
        ({"EXTNAME": "PSF_BANDS"}, "BANDS"),
        ({"HDUCLAS1": "BANDS"}, "BANDS"),
    ],
)
def test_gadf_hdu_class_key(meta, expected):
    assert gadf_hdu_class_key(meta) == expected


def test_gadf_hdu_class_key_ambiguous_extname():
    """bkg_2d and bkg_3d share EXTNAME=BACKGROUND: not guessed."""
    assert "BACKGROUND" not in GADF_EXTNAME_TO_TAG
    assert gadf_hdu_class_key({"EXTNAME": "BACKGROUND"}) not in ("BKG_2D", "BKG_3D")
    assert (
        gadf_hdu_class_key({"EXTNAME": "BACKGROUND", "HDUCLAS4": "BKG_3D"}) == "BKG_3D"
    )


def test_irf_hduclas4_uppercase():
    """Class keys are the HDUCLAS4 values as written in the specification."""
    for spec in GADF_IRF_DL3_HDU_SPECIFICATION.values():
        hduclas4 = spec["mandatory_keywords"]["HDUCLAS4"]
        assert hduclas4 == hduclas4.upper()


def test_gadf_hdu_class_key_unknown():
    assert gadf_hdu_class_key({}) == "UNKNOWN_CLAS"

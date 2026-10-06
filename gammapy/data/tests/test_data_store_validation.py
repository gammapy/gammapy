# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests of DataStore.validate and of the "format" check."""

import logging
import numpy as np
import pytest
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from gammapy.data import DataStore
from gammapy.io.fits.validation import ValidationReport
from gammapy.utils.testing import requires_data

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

TIME_META = {
    k: EVENTS_META[k] for k in ("MJDREFI", "MJDREFF", "TIMEUNIT", "TIMESYS", "TIMEREF")
}


def make_hdulist(events_meta):
    events = Table(
        {
            "EVENT_ID": np.arange(3),
            "TIME": [1.0, 2.0, 3.0] * u.s,
            "RA": [83.0, 83.5, 84.0] * u.deg,
            "DEC": [21.5, 22.0, 22.5] * u.deg,
            "ENERGY": [1.0, 2.0, 3.0] * u.TeV,
        },
        meta=events_meta,
    )
    gti = Table({"START": [0.0] * u.s, "STOP": [10.0] * u.s})
    gti.meta.update(
        HDUCLASS="GADF",
        HDUDOC=EVENTS_META["HDUDOC"],
        HDUVERS="0.3",
        HDUCLAS1="GTI",
        **TIME_META,
    )
    hdus = [fits.PrimaryHDU()]
    for table, name in [(events, "EVENTS"), (gti, "GTI")]:
        hdu = fits.table_to_hdu(table)
        hdu.name = name
        hdus.append(hdu)
    return fits.HDUList(hdus)


def make_data_store(path):
    """Data store with one compliant observation (1), one non-compliant (2),
    and one whose file is missing (3)."""
    (path / "data").mkdir()
    make_hdulist(EVENTS_META).writeto(path / "data" / "obs_1.fits")
    bad = {k: v for k, v in EVENTS_META.items() if k not in ("OBS_MODE", "DEC_PNT")}
    make_hdulist({**bad, "OBS_ID": 2}).writeto(path / "data" / "obs_2.fits")

    rows = []
    for obs_id in (1, 2, 3):
        for hdu_type, hdu_name in (("events", "EVENTS"), ("gti", "GTI")):
            rows.append(
                (obs_id, hdu_type, hdu_type, "data", f"obs_{obs_id}.fits", hdu_name)
            )
    hdu_table = Table(
        rows=rows,
        names=("OBS_ID", "HDU_TYPE", "HDU_CLASS", "FILE_DIR", "FILE_NAME", "HDU_NAME"),
    )
    hdu_table.meta.update(HDUCLASS="GADF", HDUCLAS1="INDEX", HDUCLAS2="HDU")
    hdu_table.write(path / "hdu-index.fits.gz")

    obs_table = Table({"OBS_ID": [1, 2, 3]})
    obs_table.meta.update(HDUCLASS="GADF", HDUCLAS1="INDEX", HDUCLAS2="OBS")
    obs_table.write(path / "obs-index.fits.gz")
    return path


@pytest.fixture()
def data_store_path(tmp_path):
    return make_data_store(tmp_path)


def test_from_dir_does_not_validate_by_default(data_store_path):
    data_store = DataStore.from_dir(data_store_path)
    assert data_store.validation is None


def test_from_dir_validate(data_store_path):
    data_store = DataStore.from_dir(data_store_path, validate=True)
    assert len(data_store.validation) == 6
    report = data_store.validation[(1, "events")]
    assert isinstance(report, ValidationReport)
    assert report.valid


def test_from_file_validate(data_store_path):
    hdu_index = Table.read(data_store_path / "hdu-index.fits.gz")
    hdu_index.meta["BASE_DIR"] = str(data_store_path)
    obs_index = Table.read(data_store_path / "obs-index.fits.gz")
    filename = data_store_path / "index.fits"
    hdu_index_hdu = fits.table_to_hdu(hdu_index)
    hdu_index_hdu.name = "HDU_INDEX"
    obs_index_hdu = fits.table_to_hdu(obs_index)
    obs_index_hdu.name = "OBS_INDEX"
    fits.HDUList([fits.PrimaryHDU(), hdu_index_hdu, obs_index_hdu]).writeto(filename)

    data_store = DataStore.from_file(filename, validate=True)
    assert data_store.validation[(1, "gti")].valid


def test_validate_summary(data_store_path):
    data_store = DataStore.from_dir(data_store_path)
    summary = data_store.validate()

    assert summary.colnames == [
        "OBS_ID",
        "HDU_CLASS",
        "HDU_NAME",
        "FILE",
        "VERSION",
        "HEADER",
        "TABLE",
        "N_ERRORS",
        "ERRORS",
    ]
    assert len(summary) == 6
    assert list(summary["N_ERRORS"]) == [0, 0, 2, 0, 1, 1]
    assert list(summary["HEADER"]) == ["OK", "OK", "FAIL", "OK", "FAIL", "FAIL"]
    assert "Missing mandatory keyword(s): ['OBS_MODE']" in summary["ERRORS"][2]
    assert "['DEC_PNT'] required when OBS_MODE=None" in summary["ERRORS"][2]
    assert "cannot open" in summary["ERRORS"][4]


def test_validate_summary_to_file(data_store_path):
    data_store = DataStore.from_dir(data_store_path)
    summary = data_store.validate()
    filename = data_store_path / "validation.ecsv"
    summary.write(filename)

    read = Table.read(filename)
    assert read.colnames == summary.colnames
    assert list(read["N_ERRORS"]) == list(summary["N_ERRORS"])
    # empty strings are written as missing values in ECSV
    assert list(read["ERRORS"].filled("")) == list(summary["ERRORS"])


def test_validate_logs_summary_only(data_store_path, caplog):
    data_store = DataStore.from_dir(data_store_path)
    with caplog.at_level(logging.INFO):
        data_store.validate()
    assert "GADF validation: 3/6 HDUs compliant" in caplog.text
    # per-HDU errors are not logged, they are in the summary table
    assert "Missing mandatory keyword" not in caplog.text


def test_validate_selection(data_store_path):
    data_store = DataStore.from_dir(data_store_path)

    summary = data_store.validate(obs_id=[1, 2], hdu_class="events")
    assert list(summary["OBS_ID"]) == [1, 2]
    assert set(summary["HDU_CLASS"]) == {"events"}
    assert set(data_store.validation) == {(1, "events"), (2, "events")}

    summary = data_store.validate(obs_id=1)
    assert list(summary["HDU_CLASS"]) == ["events", "gti"]


def test_validate_version(data_store_path):
    """Obs 2 lacks OBS_MODE and DEC_PNT: v0.3 needs both, v0.2 only DEC_PNT."""
    data_store = DataStore.from_dir(data_store_path)

    summary = data_store.validate(version="0.3", obs_id=2, hdu_class="events")
    assert summary["VERSION"][0] == "0.3"
    assert summary["N_ERRORS"][0] == 2

    summary = data_store.validate(version="0.2", obs_id=2, hdu_class="events")
    assert summary["VERSION"][0] == "0.2"
    assert summary["ERRORS"][0] == "header: Missing mandatory keyword(s): ['DEC_PNT']"


def test_validate_strict(data_store_path):
    data_store = DataStore.from_dir(data_store_path)
    data_store.validate(strict=True, obs_id=1)

    with pytest.raises(
        ValueError, match=r"3 HDU\(s\) not compliant with GADF"
    ) as excinfo:
        data_store.validate(strict=True)
    assert "OBS_ID=2 events" in str(excinfo.value)
    # reports are stored before raising
    assert not data_store.validation[(2, "events")].valid


def test_validate_mislabelled_hdu_class(data_store_path):
    hdu_index = Table.read(data_store_path / "hdu-index.fits.gz")
    hdu_index["HDU_CLASS"] = hdu_index["HDU_CLASS"].astype("U16")
    hdu_index["HDU_CLASS"][1] = "aeff_2d"  # obs 1 GTI declared as an effective area
    hdu_index.write(data_store_path / "hdu-index.fits.gz", overwrite=True)

    data_store = DataStore.from_dir(data_store_path)
    report = data_store.validate(obs_id=1)
    assert report["HDU_CLASS"][1] == "aeff_2d"
    assert (
        "Missing mandatory column(s): "
        "['ENERG_LO', 'ENERG_HI', 'THETA_LO', 'THETA_HI', 'EFFAREA']"
    ) in report["ERRORS"][1]


def test_check_format(data_store_path):
    data_store = DataStore.from_dir(data_store_path)
    records = list(data_store.check(checks=["format"]))
    assert len(records) == 4
    assert records[0] == {
        "level": "error",
        "obs_id": 2,
        "hdu": "events",
        "msg": "header: Missing mandatory keyword(s): ['OBS_MODE']",
    }


def test_loading_unchanged(data_store_path):
    """Validation does not interfere with the lazy loading of observations."""
    data_store = DataStore.from_dir(data_store_path, validate=True)
    obs = data_store.obs(1, required_irf="all-optional")
    assert len(obs.gti.table) == 1


@requires_data()
def test_validate_hess_dl3_dr1():
    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")

    summary = data_store.validate()
    assert len(summary) == 630
    assert set(summary["VERSION"]) == {"0.2"}
    assert (summary["N_ERRORS"] == 0).all()

    # against v0.3, the EVENTS headers fail on OBS_MODE only
    summary = data_store.validate(version="0.3", hdu_class="events")
    assert len(summary) == 105
    assert set(summary["ERRORS"]) == {
        "header: Invalid OBS_MODE='WOBBLE': allowed values "
        "['POINTING', 'RASTER', 'SLEW', 'SCAN', 'DRIFT']"
    }

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.io.fits as fits
from astropy.table import Table
from astropy.time import Time
from astropy import units as u
from astropy.utils.exceptions import AstropyDeprecationWarning
import pytest

from gammapy.utils.scripts import make_path
from gammapy.utils.testing import requires_data
from ..io import EventListReader, EventListWriter, ObservationTableReader


@requires_data()
def test_eventlist_reader_no_format():
    hess_events = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
    fermi_events = "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz"

    events = EventListReader().read(hess_events, format=None)
    assert len(events.table) == 11243

    events = EventListReader().read(fermi_events, format=None)
    assert len(events.table) == 32843


@requires_data()
def test_eventlist_reader_unkwnown_format(tmpdir):
    hess_events = make_path(
        "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
    )

    with fits.open(hess_events) as hdulist:
        hdu_events = hdulist["EVENTS"]
        hdu_events.header["HDUCLASS"] = "bad"

        hdulist.writeto(tmpdir / "tmp.fits")

    with pytest.raises(ValueError):
        EventListReader().read(tmpdir / "tmp.fits", format=None)


def test_eventlist_writer_unkwnown_format():
    with pytest.raises(ValueError):
        EventListWriter().to_hdu("tmp.fits", format="unknown")


@requires_data()
def test_eventlist_reader_empty_gadf_table():
    swgo_events = "$GAMMAPY_DATA/tests/format/swgo/map_irfs/DummyEvents.fits.gz"

    with pytest.raises(ValueError) as err:
        EventListReader().read(swgo_events, format="gadf")
    assert "ENERGY" in str(err.value)
    assert "TIME" in str(err.value)
    assert "RA" in str(err.value)
    assert "DEC" in str(err.value)


@requires_data()
def test_observationtable_reader_unknown_hdu_extension():
    hess_obs_table = make_path("$GAMMAPY_DATA/hess-dl3-dr1/obs-index.fits.gz")

    with pytest.raises(AstropyDeprecationWarning):
        obs_table = ObservationTableReader().read(
            hess_obs_table, hdu="unknown_extension"
        )
        assert len(obs_table) == 105


@requires_data()
def test_observationtable_reader_unknown_specified_format(tmpdir):
    hess_obs_table = make_path("$GAMMAPY_DATA/hess-dl3-dr1/obs-index.fits.gz")

    with fits.open(hess_obs_table) as hdulist:
        hdu_obs_table = hdulist["OBS_INDEX"]
        hdu_obs_table.header["HDUCLASS"] = "unknown-format"

        hdulist.writeto(tmpdir / "tmp.fits")

    with pytest.raises(ValueError):
        ObservationTableReader().read(tmpdir / "tmp.fits")


@requires_data()
def test_observationtable_reader_unspecified_format(tmpdir, caplog):
    hess_obs_table = make_path("$GAMMAPY_DATA/hess-dl3-dr1/obs-index.fits.gz")

    with fits.open(hess_obs_table) as hdulist:
        hdu_obs_table = hdulist["OBS_INDEX"]
        hdu_obs_table.header.remove("HDUCLASS")

        filename = "tmp.fits"
        hdulist.writeto(tmpdir / filename)

        ObservationTableReader().read(tmpdir / "tmp.fits")
        assert (
            f"Could not infer fileformat from metadata in {tmpdir / filename}, assuming GADF."
            in [_.message for _ in caplog.records]
        )


def test_observationtable_reader_gadf_converter_mandatory_keywords():
    t = Table({"RA_PNT": [1.0]}, units={"RA_PNT": u.deg})
    with pytest.raises(RuntimeError):
        ObservationTableReader._from_gadf_table(t)


def test_observationtable_reader_gadf_converter_missing_time_keywords(caplog):
    t = Table({"OBS_ID": ["1"], "TSTART": [Time("2012-01-01T00:30:00")]}, meta={})

    obs_table = ObservationTableReader._from_gadf_table(t)
    assert (
        "Found column TSTART or TSTOP in GADF table, but can not create columns in internal format due to missing header keywords in file."
        in [_.message for _ in caplog.records]
    )
    assert obs_table.keys() == ["OBS_ID"]


def test_observationtable_reader_gadf_converter_time_conversion():
    t = Table(
        {"OBS_ID": ["1"], "TSTART": [100]},
        meta={
            "MJDREFI": 50000,
            "MJDREFF": 100.0,
            "TIMEUNIT": "s",
        },
    )
    obs_table = ObservationTableReader._from_gadf_table(t)
    assert obs_table.keys() == ["OBS_ID", "TSTART"]
    assert isinstance(obs_table["TSTART"], Time)


def test_observationtable_reader_gadf_converter_invalid_time_unit(caplog):
    t = Table(
        {"OBS_ID": ["1"], "TSTART": [100]},
        meta={
            "MJDREFI": 50000,
            "MJDREFF": 100.0,
            "TIMEUNIT": "-",
        },
    )
    obs_table = ObservationTableReader._from_gadf_table(t)
    assert "Invalid unit for column TSTART." in [_.message for _ in caplog.records]
    assert obs_table.keys() == ["OBS_ID"]


def test_observationtable_reader_gadf_converter_invalid_time_datatype(caplog):
    t = Table(
        {"OBS_ID": ["1"], "TSTOP": ["-"]},
        meta={
            "MJDREFI": 50000,
            "MJDREFF": 100.0,
            "TIMEUNIT": "s",
        },
    )
    obs_table = ObservationTableReader._from_gadf_table(t)
    assert "Could not convert type for column TSTOP." in [
        _.message for _ in caplog.records
    ]
    assert obs_table.keys() == ["OBS_ID"]

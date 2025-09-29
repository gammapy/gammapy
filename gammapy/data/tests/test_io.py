# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.io.fits as fits
from astropy.table import Table
from astropy.time import Time
from astropy import units as u
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.units import UnitConversionError
import pytest
import numpy as np

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
    hess_obs_table = "$GAMMAPY_DATA/hess-dl3-dr1/obs-index.fits.gz"

    with pytest.raises(AstropyDeprecationWarning):
        obs_table = ObservationTableReader().read(
            hess_obs_table, hdu="unknown_extension"
        )
        assert len(obs_table) == 105


@requires_data()
def test_observationtable_reader_unknown_format(tmpdir):
    hess_events = make_path(
        "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
    )

    with fits.open(hess_events) as hdulist:
        hdu_events = hdulist["EVENTS"]
        hdu_events.header["HDUCLASS"] = "bad"

        hdulist.writeto(tmpdir / "tmp.fits")

    with pytest.raises(ValueError):
        ObservationTableReader().read(tmpdir / "tmp.fits")


def test_observationtable_reader_gadf_converter():
    # OBS_ID is mandatory for internal data model.
    t = Table({"RA_PNT": [1.0]}, units={"RA_PNT": u.deg})
    with pytest.raises(RuntimeError):
        ObservationTableReader._from_gadf_table(t)

    # OBS_ID is converted to int for internal model, if given as string.
    t = Table(
        {"OBS_ID": ["1"]},
        units={"OBS_ID": None},
    )
    obs_table = ObservationTableReader._from_gadf_table(t)
    assert obs_table["OBS_ID"].dtype == np.dtype(int)

    # If OBS_ID can not be converted to int, fail.
    t = Table(
        {"OBS_ID": ["-"]},
        units={"OBS_ID": None},
    )
    with pytest.raises(RuntimeError):
        obs_table = ObservationTableReader._from_gadf_table(t)

    # Unit of RA_PNT, DEC_PNT, ALT_PNT, AZ_PNT has to be deg for internal model.
    # In case of wrong dimension, warning is raised and column is dropped.
    t = Table({"OBS_ID": [1], "RA_PNT": [1.0]}, units={"OBS_ID": None, "RA_PNT": u.m})
    with pytest.raises(UnitConversionError):
        obs_table = ObservationTableReader._from_gadf_table(t)
    assert obs_table.keys() == ["OBS_ID"]

    # If TSTART or TSTOP in table but header keywords not present
    # warning is raised and time-columns are dropped.
    t = Table({"OBS_ID": ["1"], "TSTART": [Time("2012-01-01T00:30:00")]}, meta={})
    with pytest.warns(UserWarning):
        obs_table = ObservationTableReader._from_gadf_table(t)
        assert obs_table.keys() == ["OBS_ID"]
    t = Table({"OBS_ID": ["1"], "TSTOP": [Time("2012-01-01T00:30:00")]}, meta={})
    with pytest.warns(UserWarning):
        ObservationTableReader._from_gadf_table(t)
        obs_table = ObservationTableReader._from_gadf_table(t)
        assert obs_table.keys() == ["OBS_ID"]

    # If TSTART or TSTOP in table and header keywords present and correct,
    # Time is converted into TIME object.
    t = Table(
        {"OBS_ID": ["1"], "TSTART": [100]},
        meta={
            "MJDREFI": 50000,
            "MJDREFF": 100.0,
            "TIMESYS": "TT",
            "TIMEREF": "TOPOCENTER",
            "TIMEUNIT": "s",
        },
    )
    obs_table = ObservationTableReader._from_gadf_table(t)
    assert obs_table.keys() == ["OBS_ID", "TSTART"]
    assert isinstance(obs_table["TSTART"], Time) == True

    # If TSTART or TSTOP in table and header keywords present
    # but conversion fails for any reason (wrong time unit, wrong types)
    # warning is raised and time-columns are dropped.
    t = Table(
        {"OBS_ID": ["1"], "TSTART": [100]},
        meta={
            "MJDREFI": 50000,
            "MJDREFF": 100.0,
            "TIMESYS": "TT",
            "TIMEREF": "TOPOCENTER",
            "TIMEUNIT": "-",
        },
    )
    with pytest.warns(UserWarning):
        obs_table = ObservationTableReader._from_gadf_table(t)
    assert obs_table.keys() == ["OBS_ID"]
    t = Table(
        {"OBS_ID": ["1"], "TSTART": ["-"]},
        meta={
            "MJDREFI": 50000,
            "MJDREFF": 100.0,
            "TIMESYS": "TT",
            "TIMEREF": "TOPOCENTER",
            "TIMEUNIT": "s",
        },
    )
    with pytest.warns(UserWarning):
        obs_table = ObservationTableReader._from_gadf_table(t)
    assert obs_table.keys() == ["OBS_ID"]

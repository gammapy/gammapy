# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.io.fits as fits
from astropy.table import Table
from astropy.time import Time
from astropy import units as u
import pytest
from astropy.utils.exceptions import AstropyDeprecationWarning
import numpy as np

from gammapy.utils.scripts import make_path
from gammapy.utils.testing import requires_data
from ..io import EventListReader, EventListWriter, ObservationTableReader


@requires_data()
def test_observationtable_reader_unknown_hdu_extension():
    hess_obs_table = "$GAMMAPY_DATA/hess-dl3-dr1/obs-index.fits.gz"

    with pytest.raises(AstropyDeprecationWarning):
        obs_table = ObservationTableReader().read(
            hess_obs_table, hdu="unknown_extension"
        )
        assert len(obs_table) == 105


def test_observationtable_reader_gadf_converter():
    # If TSTART or TSTOP in table but header keywords not present, warning is raised.
    t = Table({"OBS_ID": ["1"], "TSTART": [Time("2012-01-01T00:30:00")]}, meta={})
    with pytest.warns(UserWarning):
        ObservationTableReader.from_gadf_table(t)

    t = Table({"OBS_ID": ["1"], "TSTOP": [Time("2012-01-01T00:30:00")]}, meta={})
    with pytest.warns(UserWarning):
        ObservationTableReader.from_gadf_table(t)

    # OBS_ID has to be of type int64 for internal model but converter ensures this.
    t_gadf = Table({"OBS_ID": ["1"], "RA_PNT": [1.0], "DEC_PNT": [1.0]})
    obs_table = ObservationTableReader.from_gadf_table(t_gadf)
    assert obs_table["OBS_ID"].dtype == np.dtype(int)

    # Unit for Column objects like RA_PNT have to be specified for internal model but converter ensures this.
    t = Table({"OBS_ID": [1], "RA_PNT": [1.0]})
    obs_table = ObservationTableReader.from_gadf_table(t)
    assert obs_table["RA_PNT"].unit == u.deg

    # Unit of RA_PNT has to be deg for internal model but converter ensures this.
    t = Table({"OBS_ID": [1], "RA_PNT": [1.0]}, units={"OBS_ID": None, "RA_PNT": u.m})
    obs_table = ObservationTableReader.from_gadf_table(t)
    assert obs_table["RA_PNT"].unit == u.deg


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

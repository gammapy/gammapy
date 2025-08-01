# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.io.fits as fits
import pytest

from gammapy.utils.scripts import make_path
from gammapy.utils.testing import requires_data
from ..io import EventListReader, EventListWriter


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

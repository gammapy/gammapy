# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.io.fits as fits
import pytest

from gammapy.utils.scripts import make_path
from ..io import EventListReader


def test_eventlist_reader_no_format():
    hess_events = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
    fermi_events = "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz"

    events = EventListReader(hess_events).read(format=None)
    assert len(events.table) == 11243

    events = EventListReader(fermi_events).read(format=None)
    assert len(events.table) == 32843


def test_eventlist_reader_unkwnown_format(tmpdir):
    hess_events = make_path(
        "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
    )

    with fits.open(hess_events) as hdulist:
        hdu_events = hdulist["EVENTS"]
        hdu_events.header["HDUCLASS"] = "bad"

        hdulist.writeto(tmpdir / "tmp.fits")

    with pytest.raises(ValueError):
        EventListReader(tmpdir / "tmp.fits").read(format=None)

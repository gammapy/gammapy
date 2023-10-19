# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from pydantic import ValidationError
from gammapy.utils.metadata import ObsInfoMetaData, PointingInfoMetaData, TargetMetaData
from gammapy.data import EventListMetaData, ObservationMetaData
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import requires_data


@pytest.fixture()
def hess_eventlist_header():
    filename = make_path(
        "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz"
    )
    hdulist = fits.open(filename)
    return hdulist["EVENTS"].header


def test_observation_metadata():
    obs_info = {
        "obs_id": 0,
        "telescope": "cta-north",
        "instrument": "lst",
        "observation_mode": "wobble",
    }
    target = {
        "name": "Crab",
        "position": SkyCoord(83.6287, 22.0147, unit="deg", frame="icrs"),
    }

    input = {
        "obs_info": ObsInfoMetaData(**obs_info),
        "pointing": PointingInfoMetaData(),
        "location": "cta_north",
        "deadtime_fraction": 0.05,
        "target": TargetMetaData(**target),
        "optional": dict(test=0.5, other=True),
    }
    meta = ObservationMetaData(**input)

    assert meta.obs_info.telescope == "cta-north"
    assert meta.obs_info.instrument == "lst"
    assert meta.obs_info.observation_mode == "wobble"
    assert_allclose(meta.location.lon.value, -17.892005)
    assert meta.target.name == "Crab"
    assert_allclose(meta.target.position.ra.deg, 83.6287)
    assert meta.optional["other"] is True

    with pytest.raises(ValidationError):
        meta.deadtime_fraction = 2.0

    with pytest.raises(ValidationError):
        meta.target.position = "J1749-2901"

    meta.target.position = None
    assert meta.target.position is None

    input_bad = input.copy()
    input_bad["location"] = "bad"

    with pytest.raises(ValueError):
        ObservationMetaData(**input_bad)


@requires_data()
def test_observation_metadata_from_header(hess_eventlist_header):
    meta = ObservationMetaData.from_header(hess_eventlist_header, format="gadf")

    assert meta.obs_info.telescope == "HESS"
    assert_allclose(meta.pointing.altaz_mean.alt.deg, 41.389789)
    assert meta.target.name == "Crab Nebula"
    assert_allclose(meta.location.lat.deg, -23.271778)
    assert "TELLIST" in meta.optional
    assert meta.optional["TELLIST"] == "1,2,3,4"


@requires_data()
def test_observation_metadata_bad(hess_eventlist_header):
    with pytest.raises(ValueError):
        ObservationMetaData.from_header(hess_eventlist_header, format="bad")

    hess_eventlist_header.pop("DEADC")
    with pytest.raises(KeyError):
        ObservationMetaData.from_header(hess_eventlist_header, format="gadf")


def test_eventlist_metadata():
    input = {
        "obs_id": "33787",
        "telescope": "HESS",
        "instrument": "H.E.S.S.",
        "observation_mode": "wobble",
        "live_time": 1534 * u.s,
        "deadtime_fraction": 0.03,
        "location": "hess",
        "optional": dict(target_name="PKS 2155-304"),
    }
    meta = EventListMetaData(**input)

    assert meta.obs_id == "33787"
    assert meta.telescope == "HESS"
    assert meta.instrument == "H.E.S.S."
    assert meta.observation_mode == "wobble"
    assert_allclose(meta.location.lat.value, -23.271777777777775)

    with pytest.raises(ValueError):
        meta.live_time = 3

    input_bad = input.copy()
    input_bad["location"] = "bad"

    with pytest.raises(ValueError):
        EventListMetaData(**input_bad)


@requires_data()
def test_eventlist_metadata_from_header(hess_eventlist_header):
    meta = EventListMetaData.from_header(hess_eventlist_header, format="gadf")

    assert meta.obs_id == "23523"
    assert meta.telescope == "HESS"
    assert meta.target_name == "Crab Nebula"
    assert_allclose(meta.location.lon.deg, 16.5002222222222)
    assert meta.live_time.value == 1581.73681640625
    assert "TASSIGN" in meta.optional
    assert meta.optional["TASSIGN"] == "Namibia"

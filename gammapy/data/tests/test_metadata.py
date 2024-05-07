# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from astropy.io import fits
from pydantic import ValidationError
from gammapy.data import EventListMetaData, ObservationMetaData
from gammapy.utils.metadata import ObsInfoMetaData, PointingInfoMetaData, TargetMetaData
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
    time_info = {
        "reference_time": "2023-01-01 00:00:00",
        "time_start": "2024-01-01 00:00:00",
        "time_stop": "2024-01-01 00:30:00",
    }

    input = {
        "obs_info": ObsInfoMetaData(**obs_info),
        "pointing": PointingInfoMetaData(),
        "location": "cta_north",
        "deadtime_fraction": 0.05,
        "time_info": time_info,
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
    assert_allclose(meta.time_info.time_stop.mjd, 60310.020833333)
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
    assert_allclose(meta.time_info.time_start.mjd, 53343.92234)
    assert_allclose(meta.time_info.time_stop.mjd, 53343.941866)
    assert meta.target.name == "Crab Nebula"
    assert_allclose(meta.location.lat.deg, -23.271778)
    assert "TELLIST" in meta.optional
    assert meta.optional["TELLIST"] == "1,2,3,4"


@requires_data()
def test_observation_metadata_bad(hess_eventlist_header):
    # TODO: adapt with proper format handling
    with pytest.raises(ValueError):
        ObservationMetaData.from_header(hess_eventlist_header, format="bad")

    hess_eventlist_header.pop("DEADC")
    with pytest.raises(KeyError):
        ObservationMetaData.from_header(hess_eventlist_header, format="gadf")


def test_eventlist_metadata():
    input = {
        "event_class": "std",
        "optional": {"DST_VER": "v1.0", "ANA_VER": "v2.2", "CAL_VER": "v1.9"},
    }

    meta = EventListMetaData(**input)

    assert meta.event_class == "std"
    assert meta.optional["DST_VER"] == "v1.0"
    assert meta.optional["ANA_VER"] == "v2.2"
    assert meta.optional["CAL_VER"] == "v1.9"

    input_bad = input.copy()
    input_bad["location"] = "bad"
    with pytest.raises(ValueError):
        EventListMetaData(**input_bad)

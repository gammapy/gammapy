# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from astropy.io import fits
from pydantic import ValidationError
from gammapy.data import ObservationMetaData
from gammapy.utils.metadata import ObsInfoMetaData, PointingInfoMetaData
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

    input = {
        "obs_info": ObsInfoMetaData(**obs_info),
        "pointing": PointingInfoMetaData(),
        "location": "cta_north",
        "deadtime_fraction": 0.05,
        "target_name": "Crab",
        "target_position": SkyCoord(83.6287, 22.0147, unit="deg", frame="icrs"),
        "optional": dict(test=0.5, other=True),
    }
    meta = ObservationMetaData(**input)

    assert meta.obs_info.telescope == "cta-north"
    assert meta.obs_info.instrument == "lst"
    assert meta.obs_info.observation_mode == "wobble"
    assert_allclose(meta.location.lon.value, -17.892005)
    assert meta.target_name == "Crab"
    assert_allclose(meta.target_position.ra.deg, 83.6287)
    assert meta.optional["other"] is True

    with pytest.raises(ValidationError):
        meta.deadtime_fraction = 2.0

    with pytest.raises(ValidationError):
        meta.target_position = "J1749-2901"

    meta.target_position = None
    assert isinstance(meta.target_position, SkyCoord)
    assert np.isnan(meta.target_position.ra.deg)

    input_bad = input.copy()
    input_bad["location"] = "bad"

    with pytest.raises(ValueError):
        ObservationMetaData(**input_bad)


@requires_data()
def test_observation_metadata_from_header(hess_eventlist_header):
    meta = ObservationMetaData.from_header(hess_eventlist_header, format="gadf")

    assert meta.obs_info.telescope == "HESS"
    assert_allclose(meta.pointing.altaz_mean.alt.deg, 41.389789)
    assert meta.target_name == "Crab Nebula"
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

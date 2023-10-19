# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from pydantic import ValidationError
from gammapy.data import ObservationMetaData


def test_observation_metadata():
    input = {
        "telescope": "cta-north",
        "instrument": "lst",
        "observation_mode": "wobble",
        "location": "cta_north",
        "deadtime_fraction": 0.05,
        "target_name": "Crab",
        "target_position": SkyCoord(83.6287, 22.0147, unit="deg", frame="icrs"),
        "optional": dict(test=0.5, other=True),
    }
    meta = ObservationMetaData(**input)

    assert meta.telescope == "cta-north"
    assert meta.instrument == "lst"
    assert meta.observation_mode == "wobble"
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

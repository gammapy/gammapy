import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from pydantic import ValidationError
from gammapy.datasets import MapDatasetMetaData


def test_mapdataset_meta_from_default():
    meta = MapDatasetMetaData.from_default()

    assert meta.creation.creator.split()[0] == "Gammapy"


def test_mapdataset_metadata():
    input = {
        "telescope": "cta-north",
        "instrument": "lst",
        "observation_mode": "wobble",
        "pointing": SkyCoord(83.6287, 22.0147, unit="deg", frame="icrs"),
        "obs_ids": 112,
        "optional": dict(test=0.5, other=True),
    }
    meta = MapDatasetMetaData(**input)

    assert meta.telescope == "cta-north"
    assert meta.instrument == "lst"
    assert meta.observation_mode == "wobble"
    assert_allclose(meta.pointing.dec.value, 22.0147)
    assert_allclose(meta.pointing.ra.deg, 83.6287)
    assert meta.obs_ids == 112
    assert meta.optional["other"] is True
    assert meta.creation.creator.split()[0] == "Gammapy"

    with pytest.raises(ValidationError):
        meta.pointing = 2.0

    with pytest.raises(ValidationError):
        meta.instrument = ["cta", "hess"]

    meta.pointing = None
    assert isinstance(meta.pointing, SkyCoord)
    assert np.isnan(meta.pointing.ra.deg)

    input_bad = input.copy()
    input_bad["obs_ids"] = "bad"

    with pytest.raises(ValueError):
        MapDatasetMetaData(**input_bad)


def test_mapdataset_metadata_lists():
    input = {
        "telescope": "cta-north",
        "instrument": "lst",
        "observation_mode": "wobble",
        "pointing": [
            SkyCoord(83.6287, 22.0147, unit="deg", frame="icrs"),
            SkyCoord(83.1287, 22.5147, unit="deg", frame="icrs"),
        ],
        "obs_ids": [111, 222],
    }
    meta = MapDatasetMetaData(**input)
    assert meta.telescope == "cta-north"
    assert meta.instrument == "lst"
    assert meta.observation_mode == "wobble"
    assert_allclose(meta.pointing[0].dec.value, 22.0147)
    assert_allclose(meta.pointing[1].ra.deg, 83.1287)
    assert meta.obs_ids == [111, 222]
    assert meta.optional is None
    assert meta.event_type == -999


def test_mapdataset_metadata_stack():
    input1 = {
        "telescope": "a",
        "instrument": "H.E.S.S.",
        "observation_mode": "wobble",
        "pointing": SkyCoord(83.6287, 22.5147, unit="deg", frame="icrs"),
        "obs_ids": 111,
        "optional": dict(test=0.5, other=True),
    }

    input2 = {
        "telescope": "b",
        "instrument": "H.E.S.S.",
        "observation_mode": "wobble",
        "pointing": SkyCoord(83.6287, 22.0147, unit="deg", frame="icrs"),
        "obs_ids": 112,
        "optional": dict(test=0.1, other=False),
    }

    meta1 = MapDatasetMetaData(**input1)
    meta2 = MapDatasetMetaData(**input2)

    meta = meta1.stack(meta2)
    assert meta.telescope == ["a", "b"]
    assert meta.instrument == "H.E.S.S."
    assert meta.observation_mode == ["wobble", "wobble"]
    assert_allclose(meta.pointing[1].dec.deg, 22.0147)
    assert meta.obs_ids == [111, 112]
    assert meta.optional["other"] == [True, False]
    assert len(meta.event_type) == 2

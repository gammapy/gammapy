import pytest
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from pydantic import ValidationError
from gammapy.datasets import MapDatasetMetaData
from gammapy.utils.metadata import ObsInfoMetaData, PointingInfoMetaData


def test_meta_default():
    meta = MapDatasetMetaData()
    assert meta.creation.creator.split()[0] == "Gammapy"
    assert meta.obs_info is None


def test_mapdataset_metadata():
    position = SkyCoord(83.6287, 22.0147, unit="deg", frame="icrs")
    obs_info_input = {
        "telescope": "cta-north",
        "instrument": "lst",
        "observation_mode": "wobble",
        "obs_id": 112,
    }
    input = {
        "obs_info": ObsInfoMetaData(**obs_info_input),
        "pointing": PointingInfoMetaData(radec_mean=position),
        "optional": dict(test=0.5, other=True),
    }
    meta = MapDatasetMetaData(**input)

    assert meta.obs_info.telescope == "cta-north"
    assert meta.obs_info.instrument == "lst"
    assert meta.obs_info.observation_mode == "wobble"
    assert_allclose(meta.pointing.radec_mean.dec.value, 22.0147)
    assert_allclose(meta.pointing.radec_mean.ra.deg, 83.6287)
    assert meta.obs_info.obs_id == 112
    assert meta.optional["other"] is True
    assert meta.creation.creator.split()[0] == "Gammapy"
    assert meta.event_type is None

    with pytest.raises(ValidationError):
        meta.pointing = 2.0

    input_bad = input.copy()
    input_bad["bad"] = position

    with pytest.raises(ValueError):
        MapDatasetMetaData(**input_bad)


def test_mapdataset_metadata_lists():
    obs_info_input1 = {
        "telescope": "cta-north",
        "instrument": "lst",
        "observation_mode": "wobble",
        "obs_id": 111,
    }
    obs_info_input2 = {
        "telescope": "cta-north",
        "instrument": "lst",
        "observation_mode": "wobble",
        "obs_id": 112,
    }
    input = {
        "obs_info": [
            ObsInfoMetaData(**obs_info_input1),
            ObsInfoMetaData(**obs_info_input2),
        ],
        "pointing": [
            PointingInfoMetaData(
                radec_mean=SkyCoord(83.6287, 22.0147, unit="deg", frame="icrs")
            ),
            PointingInfoMetaData(
                radec_mean=SkyCoord(83.1287, 22.5147, unit="deg", frame="icrs")
            ),
        ],
    }
    meta = MapDatasetMetaData(**input)
    assert meta.obs_info[0].telescope == "cta-north"
    assert meta.obs_info[0].instrument == "lst"
    assert meta.obs_info[0].observation_mode == "wobble"
    assert_allclose(meta.pointing[0].radec_mean.dec.value, 22.0147)
    assert_allclose(meta.pointing[1].radec_mean.ra.deg, 83.1287)
    assert meta.obs_info[0].obs_id == 111
    assert meta.obs_info[1].obs_id == 112
    assert meta.optional is None
    assert meta.event_type is None


def test_mapdataset_metadata_stack():
    input1 = {
        "obs_info": ObsInfoMetaData(**{"obs_id": 111}),
        "pointing": PointingInfoMetaData(
            radec_mean=SkyCoord(83.6287, 22.5147, unit="deg", frame="icrs")
        ),
        "optional": dict(test=0.5, other=True),
    }

    input2 = {
        "obs_info": ObsInfoMetaData(**{"instrument": "H.E.S.S.", "obs_id": 112}),
        "pointing": PointingInfoMetaData(
            radec_mean=SkyCoord(83.6287, 22.0147, unit="deg", frame="icrs")
        ),
        "optional": dict(test=0.1, other=False),
    }

    meta1 = MapDatasetMetaData(**input1)
    meta2 = MapDatasetMetaData(**input2)

    meta = meta1.stack(meta2)
    assert meta.creation.creator.split()[0] == "Gammapy"
    assert meta.obs_info is None

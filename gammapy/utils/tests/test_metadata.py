# Licensed under a 3-clause BSD style license - see LICENSE.rst
from typing import Optional
import pytest
from numpy.testing import assert_allclose
from astropy.coordinates import AltAz, SkyCoord
from astropy.io import fits
from pydantic import ValidationError
from gammapy.utils.metadata import (
    CreatorMetaData,
    MetaData,
    ObsInfoMetaData,
    PointingInfoMetaData,
)
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import requires_data


@pytest.fixture()
def hess_eventlist_header():
    filename = make_path(
        "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz"
    )
    hdulist = fits.open(filename)
    return hdulist["EVENTS"].header


def test_creator():
    default = CreatorMetaData(date="2022-01-01", creator="gammapy", origin="CTA")

    assert default.creator == "gammapy"
    assert default.origin == "CTA"
    assert_allclose(default.date.mjd, 59580)

    default.creator = "other"
    assert default.creator == "other"

    with pytest.raises(ValidationError):
        default.date = 3


def test_subclass():
    class TestMetaData(MetaData):
        name: str
        mode: Optional[SkyCoord]
        creation: Optional[CreatorMetaData]

    creator = CreatorMetaData.from_default()
    test_meta = TestMetaData(name="test", creation=creator)

    assert test_meta.name == "test"
    assert test_meta.creation.creator.split()[0] == "Gammapy"

    with pytest.raises(ValidationError):
        test_meta.mode = "coord"

    yaml_str = test_meta.to_yaml()
    assert "name: test" in yaml_str
    assert "creation:" in yaml_str

    test_meta.extra = 3
    assert test_meta.extra == 3


def test_obs_info():
    obs_info = ObsInfoMetaData(obs_id="23523")

    assert obs_info.telescope is None
    assert obs_info.obs_id == "23523"

    obs_info.obs_id = 23523
    assert obs_info.obs_id == "23523"

    obs_info.instrument = "CTA-North"
    assert obs_info.instrument == "CTA-North"


def test_pointing_info():
    position = SkyCoord(83.6287, 22.0147, unit="deg", frame="icrs")
    altaz = AltAz("20 deg", "45 deg")

    pointing = PointingInfoMetaData(radec_mean=position, altaz_mean=altaz)

    assert isinstance(pointing.altaz_mean, SkyCoord)
    assert_allclose(pointing.altaz_mean.alt.deg, 45.0)

    assert_allclose(pointing.radec_mean.ra.deg, 83.6287)

    pointing.radec_mean = position.galactic
    assert_allclose(pointing.radec_mean.ra.deg, 83.6287)

    with pytest.raises(ValidationError):
        pointing.radec_mean = altaz


@requires_data()
def test_obs_info_from_header(hess_eventlist_header):
    meta = ObsInfoMetaData.from_header(hess_eventlist_header, format="gadf")

    assert meta.telescope == "HESS"
    assert meta.obs_id == "23523"
    assert meta.observation_mode == "WOBBLE"
    assert meta.sub_array is None

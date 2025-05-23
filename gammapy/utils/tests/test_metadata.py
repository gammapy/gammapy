# Licensed under a 3-clause BSD style license - see LICENSE.rst
from typing import ClassVar, Literal, Optional
import pytest
from numpy.testing import assert_allclose
from astropy.coordinates import AltAz, SkyCoord
from astropy.io import fits
from astropy.time import Time
from pydantic import ValidationError
from gammapy.utils.metadata import (
    METADATA_FITS_KEYS,
    CreatorMetaData,
    MetaData,
    ObsInfoMetaData,
    PointingInfoMetaData,
    TargetMetaData,
    TimeInfoMetaData,
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


def test_creator_to_header():
    header = CreatorMetaData(
        date="2022-01-01", creator="gammapy", origin="CTA"
    ).to_header(format="gadf")

    assert header["CREATOR"] == "gammapy"
    assert header["ORIGIN"] == "CTA"
    assert header["CREATED"] == "2022-01-01 00:00:00.000"


def test_creator_from_incorrect_header():
    # Create header with a 'bad' date
    hdu = fits.PrimaryHDU()
    hdu.header["CREATOR"] = "gammapy"
    hdu.header["CREATED"] = "Tues 6 Feb"

    meta = CreatorMetaData.from_header(hdu.header)

    assert meta.date == hdu.header["CREATED"]
    assert meta.creator == hdu.header["CREATOR"]


def test_subclass():
    class TestMetaData(MetaData):
        _tag: ClassVar[Literal["tag"]] = "tag"
        name: str
        mode: Optional[SkyCoord] = None
        creation: Optional[CreatorMetaData] = None

    creator = CreatorMetaData()
    test_meta = TestMetaData(name="test", creation=creator)

    assert test_meta.tag == "tag"

    assert test_meta.name == "test"
    assert test_meta.creation.creator.split()[0] == "Gammapy"

    with pytest.raises(ValidationError):
        test_meta.mode = "coord"

    yaml_str = test_meta.to_yaml()
    assert "name: test" in yaml_str
    assert "creation:" in yaml_str


def test_obs_info():
    obs_info = ObsInfoMetaData(obs_id="23523")

    assert obs_info.telescope is None
    assert obs_info.obs_id == 23523

    obs_info.obs_id = 23523
    assert obs_info.obs_id == 23523

    with pytest.raises(ValidationError):
        obs_info.obs_id = "ab"

    obs_info.instrument = "CTA-North"
    assert obs_info.instrument == "CTA-North"


@requires_data()
def test_obs_info_from_header(hess_eventlist_header):
    meta = ObsInfoMetaData.from_header(hess_eventlist_header, format="gadf")

    assert meta.telescope == "HESS"
    assert meta.obs_id == 23523
    assert meta.observation_mode == "WOBBLE"
    assert meta.sub_array is None


def test_obs_info_to_header():
    obs_info = ObsInfoMetaData(obs_id=23523, telescope="CTA-South")

    header = obs_info.to_header("gadf")

    assert header["OBS_ID"] == 23523
    assert header["TELESCOP"] == "CTA-South"
    assert "OBS_MODE" not in header


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


def test_pointing_info_to_header():
    position = SkyCoord(83.6287, 22.0147, unit="deg", frame="icrs")
    altaz = AltAz("20 deg", "45 deg")

    header = PointingInfoMetaData(radec_mean=position, altaz_mean=altaz).to_header(
        "gadf"
    )

    assert_allclose(header["RA_PNT"], 83.6287)
    assert_allclose(header["AZ_PNT"], 20.0)

    header = PointingInfoMetaData(radec_mean=position).to_header("gadf")
    assert "AZ_PNT" not in header.keys()

    with pytest.raises(ValueError):
        PointingInfoMetaData(radec_mean=position, altaz_mean=altaz).to_header("bad")


@requires_data()
def test_pointing_info_from_header(hess_eventlist_header):
    meta = PointingInfoMetaData.from_header(hess_eventlist_header, format="gadf")

    assert_allclose(meta.radec_mean.ra.deg, 83.633333)
    assert_allclose(meta.altaz_mean.alt.deg, 41.389789)

    meta = PointingInfoMetaData.from_header({})
    assert meta.altaz_mean is None
    assert meta.radec_mean is None


def test_target_metadata():
    meta = TargetMetaData(
        name="center", position=SkyCoord(0.0, 0.0, unit="deg", frame="galactic")
    )
    header = meta.to_header(format="gadf")

    assert meta.name == "center"
    assert_allclose(meta.position.ra.deg, 266.404988)

    assert header["OBJECT"] == "center"
    assert_allclose(header["RA_OBJ"], 266.404988)

    header = TargetMetaData(name="center").to_header("gadf")
    assert header["OBJECT"] == "center"
    assert "RA_OBJ" not in header.keys()


@requires_data()
def test_target_metadata_from_header(hess_eventlist_header):
    meta = TargetMetaData.from_header(hess_eventlist_header, format="gadf")

    assert meta.name == "Crab Nebula"
    assert_allclose(meta.position.ra.deg, 83.63333333)


def test_time_info_metadata():
    meta = TimeInfoMetaData(
        reference_time="2023-01-01 00:00:00",
        time_start="2024-01-01 00:00:00",
        time_stop="2024-01-01 12:00:00",
    )

    assert isinstance(meta.reference_time, Time)
    delta = meta.time_stop - meta.time_start
    assert_allclose(delta.to_value("h"), 12)

    header = meta.to_header(format="gadf")
    assert header["MJDREFI"] == 59945
    assert header["TIMESYS"] == "tt"
    assert_allclose(header["TSTART"], 31536000.000000257)


@requires_data()
def test_time_info_metadata_from_header(hess_eventlist_header):
    meta = TimeInfoMetaData.from_header(hess_eventlist_header, format="gadf")

    assert_allclose(meta.reference_time.mjd, 51910.00074287037)
    assert_allclose(meta.time_start.mjd, 53343.92234009259)


def test_subclass_to_from_header():
    class TestMetaData(MetaData):
        _tag: ClassVar[Literal["test"]] = "test"
        creation: Optional[CreatorMetaData]
        pointing: Optional[PointingInfoMetaData]

    METADATA_FITS_KEYS.update({"test": {}})

    creator = CreatorMetaData(date="2022-01-01", creator="gammapy", origin="CTA")
    position = SkyCoord(83.6287, 22.0147, unit="deg", frame="icrs")
    altaz = AltAz("20 deg", "45 deg")

    pointing = PointingInfoMetaData(radec_mean=position, altaz_mean=altaz)

    test_meta = TestMetaData(pointing=pointing, creation=creator)

    header = test_meta.to_header()

    assert header["CREATOR"] == "gammapy"
    assert header["ORIGIN"] == "CTA"
    assert_allclose(header["RA_PNT"], 83.6287)
    assert_allclose(header["AZ_PNT"], 20.0)

    new = TestMetaData.from_header(header)

    assert new.creation.creator == "gammapy"
    assert_allclose(new.creation.date.decimalyear, 2022)
    assert_allclose(new.pointing.radec_mean.ra.deg, 83.6287)
    assert_allclose(new.pointing.altaz_mean.alt.deg, 45)
    # no new attributes allowed
    with pytest.raises(ValidationError):
        test_meta.extra = 3

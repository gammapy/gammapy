# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from pydantic import ValidationError
from gammapy.utils.metadata import CreatorMetaData, TargetMetaData
from gammapy.utils.testing import requires_data
from gammapy.version import version
from ..metadata import FluxMetaData


@pytest.fixture()
def default():
    default = FluxMetaData(
        sed_type="likelihood",
        creation=CreatorMetaData(
            date="2022-01-01", creator="gammapy test", origin="CTA"
        ),
        target=TargetMetaData(
            name="PKS2155-304",
            position=SkyCoord(
                "21 58 52.06 -30 13 32.11", frame="icrs", unit=(u.hourangle, u.deg)
            ),
        ),
        n_sigma=2.0,
        n_sigma_ul=None,
    )
    return default


@requires_data()
def test_creator(default):

    assert default.creation.creator == "gammapy test"
    assert default.sed_type == "likelihood"
    assert default.sed_type_init is None
    assert default.target.name == "PKS2155-304"
    assert default.target.position is not None

    with pytest.raises(ValidationError):
        default.target.position = 2.0

    with pytest.raises(ValidationError):
        default.sed_type = "test"

    default = FluxMetaData()
    assert default.creation.creator == f"Gammapy {version}"


def test_from_header():
    tdict = {
        "SED_TYPE": "dnde",
        "N_SIGMA": "4.3",
        "OBJECT": "RXJ",
        "RA_OBJ": "123.5",
        "DEC_OBJ": "-4.8",
        "OBS_IDS": ["1", "2", "3", "6"],
        "DATASETS": "myanalysis",
        "INSTRU": "HAWC",
        "CREATED": "2023-10-27 16:05:09.795",
        "ORIGIN": "Gammapy v2.3",
        "CREATOR": "mynotebook",
        "EPHEM": "fermi23.1",
    }

    meta = FluxMetaData.from_header(tdict)
    assert meta.n_sigma == 4.3
    assert meta.creation.origin == "Gammapy v2.3"
    assert meta.target.position == SkyCoord(123.5 * u.deg, -4.8 * u.deg, frame="icrs")
    assert meta.optional is None
    # TODO: add for support: assert meta.optional["EPHEM"] == "fermi23.1"


@requires_data()
def test_to_header(default):
    hdr = default.to_header()
    assert hdr["OBJECT"] == "PKS2155-304"
    assert hdr["ORIGIN"] == "CTA"
    assert hdr["RA_OBJ"] == 329.7169166666666
    assert hdr["N_SIGMA"] == 2.0
    assert hdr["CREATOR"] == "gammapy test"

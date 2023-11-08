# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from pydantic import ValidationError
from gammapy.utils.metadata import CreatorMetaData, TargetMetaData
from gammapy.utils.testing import requires_data
from gammapy.version import version
from ..metadata import FluxMetaData


@pytest.fixture()
def default():
    default = FluxMetaData(
        creation=CreatorMetaData(date="2022-01-01", creator="gammapy", origin="CTA"),
        instrument="CTAS",
        n_sigma=2.0,
        obs_ids=[1, 2, 3],
        dataset_names=["aa", "tt"],
        n_sigma_ul=None,
    )
    return default


@requires_data()
def test_creator(default):

    assert default.creation.creator == "gammapy"
    assert default.dataset_names[1] == "tt"
    assert default.sed_type_init is None

    target = TargetMetaData(name="PKS2155", position=None)
    default.target = target
    assert np.isnan(default.target.position.ra)

    with pytest.raises(ValidationError):
        default.obs_ids = 4.2

    with pytest.raises(ValidationError):
        default.target.position = 2.0

    with pytest.raises(ValidationError):
        default.sed_type = "test"

    default = FluxMetaData.from_default()
    assert default.instrument is None
    assert default.creation.creator == f"Gammapy {version}"


def test_from_fits_header():
    tdict = {
        "SED_TYPE": "dnde",
        "N_SIGMA": "4.3",
        "TARGETNA": "RXJ",
        "RA_OBJ": "123.5",
        "DEC_OBJ": "-4.8",
        "OBS_IDS": "1 2 3 6",
        "DATASETS": "myanalysis",
        "INSTRU": "HAWC",
        "CREATED": "2023-10-27 16:05:09.795",
        "ORIGIN": "Gammapy v2.3",
        "CREATOR": "mynotebook",
        "EPHEM": "fermi23.1",
    }

    meta = FluxMetaData.from_header(tdict)
    print(meta)
    assert meta.n_sigma == 4.3
    assert meta.optional["ephem"] == "fermi23.1"
    assert meta.target_position == SkyCoord(123.5 * u.deg, -4.8 * u.deg, frame="icrs")


@requires_data()
def test_to_table(default):
    table = Table()
    default.to_table(table)
    print(table.meta)
    assert table.meta["OBS_IDS"] == "1 2 3"
    assert table.meta["ORIGIN"] == "CTA"
    assert table.meta["RA_OBJ"] == "83.633000"
    assert table.meta["N_SIGMA"] == "2.0"

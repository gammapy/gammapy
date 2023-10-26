# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from pydantic import ValidationError
from gammapy.utils.metadata import CreatorMetaData
from gammapy.version import version
from ..metadata import FluxMetaData


def test_creator():
    default = FluxMetaData(
        creation=CreatorMetaData(date="2022-01-01", creator="gammapy", origin="CTA"),
        instrument="CTAS",
        target_position=SkyCoord(83.633 * u.deg, 22.014 * u.deg, frame="icrs"),
        n_sigma=2.0,
        obs_ids=[1, 2, 3],
        dataset_names=["aa", "tt"],
        n_sigma_ul=None,
    )

    assert default.creation.creator == "gammapy"
    assert default.dataset_names[1] == "tt"
    assert default.sed_type_init is None

    default.target_position = None
    assert np.isnan(default.target_position.ra)

    with pytest.raises(ValidationError):
        default.obs_ids = 4.2
    with pytest.raises(ValidationError):
        default.obs_ids = [5, "bad"]

    with pytest.raises(ValidationError):
        default.target_position = 2.0

    default = FluxMetaData.from_default()
    assert default.instrument is None
    assert default.creation.creator == f"Gammapy {version}"

    default = FluxMetaData(
        creation=CreatorMetaData.from_default(),
        n_sigma=None,
        obs_ids=None,
        dataset_names=None,
    )

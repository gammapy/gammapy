# Licensed under a 3-clause BSD style license - see LICENSE.rst
from typing import Optional
import pytest
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from pydantic import ValidationError
from gammapy.utils.metadata import CreatorMetaData, MetaData


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

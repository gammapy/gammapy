# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from pydantic import ValidationError
from gammapy.utils.metadata import CreatorMetaData


def test_creator():
    default = CreatorMetaData(date="2022-01-01", creator="gammapy", origin="CTA")

    assert default.creator == "gammapy"
    assert default.origin == "CTA"
    assert_allclose(default.date.mjd, 59580)

    default.creator = "other"
    assert default.creator == "other"

    with pytest.raises(ValidationError):
        default.date = 3

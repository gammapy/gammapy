# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from ...utils.testing import requires_data
from ..core import gammapy_data


@requires_data("gammapy-data")
def test_gammapy_data():
    """Try loading a file from gammapy-data.
    """
    assert gammapy_data.dir.is_dir()

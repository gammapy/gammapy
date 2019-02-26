# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from ...utils.testing import requires_data
from ..core import gammapy_data
from ...datasets import load_poisson_stats_image


@requires_data("gammapy-data")
def test_gammapy_data():
    """Try loading a file from gammapy-data.
    """
    assert gammapy_data.dir.is_dir()


@requires_data("gammapy-data")
def test_load_poisson_stats_image():
    data = load_poisson_stats_image()
    assert data.sum() == 40896

    images = load_poisson_stats_image(extra_info=True)
    refs = dict(counts=40896, model=41000, source=1000, background=40000)
    for name, expected in refs.items():
        assert_allclose(images[name].sum(), expected)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from ...utils.testing import requires_data
from ..core import gammapy_extra
from ...datasets import load_poisson_stats_image


@requires_data("gammapy-extra")
def test_gammapy_extra():
    """Try loading a file from gammapy-extra.
    """
    assert gammapy_extra.dir.is_dir()


@requires_data("gammapy-extra")
def test_load_poisson_stats_image():
    data = load_poisson_stats_image()
    assert data.sum() == 40896

    images = load_poisson_stats_image(extra_info=True)
    refs = dict(counts=40896, model=41000, source=1000, background=40000)
    for name, expected in refs.items():
        assert_allclose(images[name].sum(), expected)

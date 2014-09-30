# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from numpy.testing import assert_allclose
from ...utils.testing import assert_quantity
from ...datasets import (load_poisson_stats_image,
                         load_arf_fits_table,
                         load_psf_fits_table,
                         load_atnf_sample,
                         load_tevcat,
                         )


def test_load_poisson_stats_image():
    """Get the data file via the gammapy.data.poisson_stats_image function"""
    data = load_poisson_stats_image()
    assert data.sum() == 40896

    images = load_poisson_stats_image(extra_info=True)
    refs = dict(counts=40896, model=41000, source=1000, background=40000)
    for name, expected in refs.items():
        assert_allclose(images[name].sum(), expected)


def test_load_arf_fits_table():
    data = load_arf_fits_table()
    assert len(data) == 2


def test_load_psf_fits_table():
    data = load_psf_fits_table()
    assert len(data) == 2


def test_load_atnf_sample():
    data = load_atnf_sample()
    assert len(data) == 10


def test_load_tevcat():
    table = load_tevcat()
    assert len(table) == 173

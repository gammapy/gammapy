# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import remote_data
from ...datasets import (
    load_catalog_atnf,
    load_catalog_green,
    load_catalog_snrcat,
    load_catalog_tevcat,
)


def test_load_catalog_atnf():
    catalog = load_catalog_atnf(small_sample=True)
    assert len(catalog) == 10


@remote_data
def test_load_catalog_green():
    catalog = load_catalog_green()
    assert len(catalog) == 173


@remote_data
def test_load_catalog_snrcat():
    catalog = load_catalog_snrcat()
    assert len(catalog) == 173


@remote_data
def test_load_catalog_tevcat():
    catalog = load_catalog_tevcat()
    assert len(catalog) == 173

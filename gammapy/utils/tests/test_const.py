# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from ..const import conversion_factor as cf


def test_conversion_factor():
    assert_allclose(cf('erg', 'GeV'), 624.150947961)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ...image import make_header


def test_make_header():
    header = make_header()
    assert header['NAXIS'] == 2
    assert header['NAXIS1'] == 100
    assert header['NAXIS2'] == 100
    assert header['CTYPE1'] == 'GLON-CAR'
    assert header['CTYPE2'] == 'GLAT-CAR'

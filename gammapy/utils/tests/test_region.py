# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from ..region import make_ds9_region


def test_make_ds9_region():
    source = dict(Type='Gaussian', GLON=42, GLAT=43.2, Sigma=99)
    attrs = dict(text='Anna')
    expected = 'galactic;circle(42,43.2,297.0) # text={Anna}\n'
    actual = make_ds9_region(source, attrs=attrs)
    assert actual == expected

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ..units import standardise_unit


def test_standardise_unit():
    assert standardise_unit("ph cm-2 s-1") == "cm-2 s-1"
    assert standardise_unit("ct cm-2 s-1") == "cm-2 s-1"
    assert standardise_unit("cm-2 s-1") == "cm-2 s-1"

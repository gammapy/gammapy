# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.utils.units import standardise_unit


def test_standardise_unit():
    assert standardise_unit("ph cm-2 s-1") == "cm-2 s-1"
    assert standardise_unit("ct cm-2 s-1") == "cm-2 s-1"
    assert standardise_unit("cm-2 s-1") == "cm-2 s-1"

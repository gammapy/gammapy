# Licensed under a 3-clause BSD style license - see LICENSE.rst
from ..cwt import CWT

def test_CWT():
    cwt = CWT(nscales=6, min_scale=6.0, scale_step=1.3)

    # TODO: run on test data
    assert 42 == 42

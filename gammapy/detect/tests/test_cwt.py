# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ...utils.testing import requires_dependency
from ...detect import CWT


@requires_dependency('scipy')
def test_CWT():
    cwt = CWT(nscales=6, min_scale=6.0, scale_step=1.3)

    # TODO: run on test data
    assert 42 == 42

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ...testing import requires_dependency


@requires_dependency('sherpa')
def test_sherpa_wrapper():
    from ..sherpa import SHERPA_OPTMETHODS
    assert 'levmar' in SHERPA_OPTMETHODS

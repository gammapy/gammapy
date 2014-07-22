# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from ..array import array_stats_str


def test_array_stats_str():
    actual = array_stats_str(np.pi, 'pi')
    assert actual == 'pi             : size =     1, min =  3.142, max =  3.142\n'

    actual = array_stats_str([np.pi, 42])
    assert actual == 'size =     2, min =  3.142, max = 42.000\n'

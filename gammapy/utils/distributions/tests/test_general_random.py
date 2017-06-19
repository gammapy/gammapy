# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pytest
from ....utils.distributions import GeneralRandom


@pytest.mark.xfail
def test_simple(make_plots=False):
    """Show the member vectors for a very simple example
    to better understand how exacly this lookup works."""
    # Define the example you want to investigate:
    r1 = GeneralRandom(np.arange(10), np.ones(10), 100)
    r2 = GeneralRandom(np.arange(5), np.ones(5), 20)

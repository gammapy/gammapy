# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from ...datasets import get_path
from ...data import GoodTimeIntervals

filename = get_path('hess/run_0023037_hard_eventlist.fits.gz')


def test_GoodTimeIntervals():
    gtis = GoodTimeIntervals.read(filename, hdu='GTI')

    assert len(gtis) == 1
    assert 'Good time interval info' in gtis.info
    gtis.time_observations
    gtis.time_observation
    gtis.time_start
    gtis.time_stop
    gtis.time_dead_fraction
    gtis.time_live

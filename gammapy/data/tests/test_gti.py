# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ...utils.testing import requires_data
from ...datasets import gammapy_extra
from ...data import GoodTimeIntervals


@requires_data('gammapy-extra')
def test_GoodTimeIntervals():
    filename = gammapy_extra.filename('test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz')
    gtis = GoodTimeIntervals.read(filename, hdu='GTI')

    assert len(gtis) == 1
    assert 'Good time interval (GTI) info' in gtis.summary
    assert '{:1.5f}'.format(gtis.time_delta[0]) == '1568.00000 s'
    assert '{:1.5f}'.format(gtis.time_sum) == '1568.00000 s'
    assert gtis.time_start[0].iso == '2004-10-14 00:08:32.000'
    assert gtis.time_stop[-1].iso == '2004-10-14 00:34:40.000'

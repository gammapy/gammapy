# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ...utils.testing import requires_data
from ...data import GTI


@requires_data("gammapy-extra")
def test_gti_hess():
    filename = (
        "$GAMMAPY_EXTRA/test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz"
    )
    gti = GTI.read(filename)
    assert "GTI" in str(gti)

    assert len(gti.table) == 1
    assert "{:1.5f}".format(gti.time_delta[0]) == "1568.00000 s"
    assert "{:1.5f}".format(gti.time_sum) == "1568.00000 s"
    assert gti.time_start[0].iso == "2004-10-14 00:08:32.000"
    assert gti.time_stop[-1].iso == "2004-10-14 00:34:40.000"


@requires_data("gammapy-extra")
def test_gti_fermi():
    filename = "$GAMMAPY_EXTRA/datasets/fermi_2fhl/2fhl_events.fits.gz"
    gti = GTI.read(filename)
    assert "GTI" in str(gti)

    assert len(gti.table) == 36589
    assert "{:1.5f}".format(gti.time_delta[0]) == "352.49307 s"
    assert "{:1.5f}".format(gti.time_sum) == "171273490.97510 s"
    assert gti.time_start[0].fits == "2008-08-04T15:49:40.784(TT)"
    assert gti.time_stop[-1].fits == "2015-01-31T23:50:42.784(TT)"

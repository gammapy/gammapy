# Licensed under a 3-clause BSD style license - see LICENSE.rst
from ..io import EventListReader


def test_eventlist_reader():
    hess_events = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
    fermi_events = "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz"

    events = EventListReader(hess_events).read()
    assert len(events.table) == 11243

    events = EventListReader(fermi_events).read()
    assert len(events.table) == 32843

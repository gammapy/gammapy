"""Eventlist coordinate check.
"""
from gammapy.data import EventListDataset

filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz'
event_list = EventListDataset.read(filename)
print(event_list.info)
event_list.check()

"""
TODO: figure out the origin of this offset:
ALT / AZ not consistent with RA / DEC. Max separation: 726.6134257108188 arcsec
"""

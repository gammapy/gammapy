"""Eventlist coordinate check.
"""
from gammapy.data import EventListDataset
from gammapy.datasets import get_path


filename = get_path('hess/run_0023037_hard_eventlist.fits.gz')
event_list = EventListDataset.read(filename)
print(event_list.info)
event_list.check()

"""
TODO: figure out the origin of this offset:

ALT / AZ not consistent with RA / DEC. Max separation: 823.3407076612169 arcsec
"""
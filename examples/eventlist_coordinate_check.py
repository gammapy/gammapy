"""Eventlist coordinate check.
"""
from gammapy.data import EventListDataset, check_event_list_coordinates
from gammapy.datasets import get_path


filename = get_path('hess/run_0023037_hard_eventlist.fits.gz')
event_list = EventListDataset.read(filename)
print(event_list.info)
# check_event_list_coordinates(event_list)

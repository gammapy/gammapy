"""Eventlist coordinate check.
"""
from gammapy.data import EventList, check_event_list_coordinates

filename = '/Users/deil/work/host/howto/ctools_crab/cta-1dc/data/hess/CTA1DC-HESS-run_00023523_eventlist.fits'
event_list = EventList.read(filename)
check_event_list_coordinates(event_list)

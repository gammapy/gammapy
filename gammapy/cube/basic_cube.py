# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to perform basic functions for map and cube analysis.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from ..maps import WcsNDMap


def fill_map_counts(event_list, ndmap):
    """Fill a ``WcsNDMap` with events from an EventList.

     The energy of the events is used for a non-spatial axis homogeneous to energy.
     The other non-spatial axis names should have an entry in the colum names of the ``EventList``

     Parameters
     ----------
     event_list : `~gammapy.data.EventList`
             the input event list
     nd_map : `~gammapy.maps.WcsNDMap`
         Target map
     """
    # The list will contain the event table entries to be fed into the WcsNDMap
    tmp = dict()
    ref_geom = ndmap.geom

    # Add sky coordinates to dictionary
    tmp.update(skycoord=event_list.radec)

    # No check the other axes and find corresponding entries in the EventList
    # energy and time are specific types
    # TODO: add proper extraction for time
    for i, axis in enumerate(ref_geom.axes):
        if axis.type == 'energy':
            # This axis is the energy
            tmp.update({axis.name: event_list.energy.to(axis.unit)})
        elif axis.name.upper() in event_list.table.colnames:
            # Here we assume that colnames are all capital
            tmp.update({axis.name: event_list.table[axis.name.upper()].to(axis.unit)})
        elif axis.name.lower() in event_list.table.colnames:
            # Here we assume that colnames are all lower cases
            tmp.update({axis.name: event_list.table[axis.name.lower()].to(axis.unit)})
        else:
            raise ValueError("Cannot find MapGeom axis {} in EventList", axis.name)

    # Fill it
    ndmap.fill_by_coord(tmp)


def make_map_counts(event_list, ref_geom):
    """Build a ``WcsNDMap` with events from an EventList.

    The energy of the events is used for a non-spatial axis homogeneous to energy.
    The other non-spatial axis names should have an entry in the colum names of the ``EventList``

    Parameters
    ----------
    event_list : `~gammapy.data.EventList`
            the input event list
    ref_geom : `~gammapy.maps.WcsGeom`
            the reference geometry

    Returns
    -------
    cntmap : `~gammapy.maps.WcsNDMap`
        Count cube (3D) in true energy bins
    """
    # Create map
    cntmap = WcsNDMap(ref_geom)
    # Fill it
    fill_map_counts(event_list, cntmap)

    return cntmap

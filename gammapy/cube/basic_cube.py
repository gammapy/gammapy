# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to perform basic functions for map and cube analysis.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from ..maps import WcsNDMap

def make_map_counts(evts, valid_ref_map):
    """
    Build a ``WcsNDMap` with events from an EventList.
    The energy of the events is used for a non-spatial axis homogeneous to energy.
    The other non-spatial axis names should have an entry in the colum names of the ``EventList``

    Parameters
    __________

    evts : `~gammapy.data.EventList`
            the input event list
    valid_ref_map : `~gammapy.maps.WcsNDMap`
        Map containing the valid pixels

    Returns
    -------
    cntmap : `~gammapy.maps.WcsNDMap`
        Count cube (3D) in true energy bins

    """
    # The list will contain the event table entries to be fed into the WcsNDMap
    tmp = list()
    ref_geom = valid_ref_map.geom
    # Convert events coordinates
    if ref_geom.coordsys == 'GAL':
        tmp.append(evts.galactic.l)
        tmp.append(evts.galactic.b)
    elif ref_geom.coordsys == 'CEL':
        tmp.append(evts.radec.ra)
        tmp.append(evts.radec.dec)
    else:
        # should raise an error here. The map is not correct.
        raise ValueError("Incorrect coordsys of input map.")

    for i, axis in enumerate(ref_geom.axes):
        if axis.unit.is_equivalent("eV"):
            # This axis is the energy
            tmp.append(evts.energy.to(axis.unit))
        elif axis.name.upper() in evts.table.colnames:
            # Here we assume that colanmes are all capital
            tmp.append(evts.table[axis.name.upper()].to(axis.unit))
        elif axis.name.lower() in evts.table.colnames:
            # Here we assume that colanmes are not capital
            tmp.append(evts.table[axis.name.lower()].to(axis.unit))
        else:
            raise ValueError("Cannot find MapGeom axis {} in EventList", axis.name)

    # Create map
    cntmap = WcsNDMap(ref_geom)
    # Fill it
    cntmap.fill_by_coords(tmp)

    # Put counts outside validity region to zero
    cntmap.data *= valid_ref_map.data

    return cntmap

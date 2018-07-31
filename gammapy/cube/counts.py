# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to perform basic functions for map and cube analysis.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from ..maps import WcsNDMap

__all__ = [
    'fill_map_counts',
    'make_map_counts',
]


def fill_map_counts(count_map, event_list):
    """Fill events into a counts map.

    The energy of the events is used for a non-spatial axis homogeneous to energy.
    The other non-spatial axis names should have an entry in the column names of the ``EventList``

    Parameters
    ----------
    count_map : `~gammapy.maps.Map`
        Map object, will be filled by this function.
    event_list : `~gammapy.data.EventList`
        Event list
    """
    geom = count_map.geom

    # Make a coordinate dictionary; skycoord is always added
    coord_dict = dict(skycoord=event_list.radec)

    # Now add one coordinate for each extra map axis
    for axis in geom.axes:
        if axis.type == 'energy':
            # This axis is the energy. We treat it differently because axis.name could be e.g. 'energy_reco'
            coord_dict[axis.name] = event_list.energy.to(axis.unit)
        # TODO: add proper extraction for time
        else:
            # We look for other axes name in the table column names (case insensitive)
            colnames = [_.upper() for _ in event_list.table.colnames]
            if axis.name.upper() in colnames:
                column_name = event_list.table.colnames[colnames.index(axis.name.upper())]
                coord_dict.update({axis.name: event_list.table[column_name].to(axis.unit)})
            else:
                raise ValueError("Cannot find MapGeom axis {!r} in EventList".format(axis.name))

    count_map.fill_by_coord(coord_dict)


def make_map_counts(events, ref_geom, pointing, offset_max):
    """Build a WcsNDMap (space - energy) with events from an EventList.

    The energy of the events is used for the non-spatial axis.

    Parameters
    ----------
    events : `~gammapy.data.EventList`
        Event list
    ref_geom : `~gammapy.maps.WcsGeom`
        Reference WcsGeom object used to define geometry (space - energy)
    pointing : `~astropy.coordinates.SkyCoord`
        Pointing direction
    offset_max : `~astropy.coordinates.Angle`
        Maximum field of view offset.

    Returns
    -------
    cntmap : `~gammapy.maps.WcsNDMap`
        Count cube (3D) in true energy bins
    """
    counts_map = WcsNDMap(ref_geom)
    fill_map_counts(counts_map, events)

    # Compute and apply FOV offset mask
    offset = ref_geom.separation(pointing)
    offset_mask = offset >= offset_max
    counts_map.data[:, offset_mask] = 0

    return counts_map

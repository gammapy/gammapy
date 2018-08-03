# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.units import Quantity

__all__ = [
    'fill_map_counts',
]


def fill_map_counts(counts_map, events):
    """Fill events into a counts map.

    The energy of the events is used for a non-spatial axis homogeneous to energy.
    The other non-spatial axis names should have an entry in the column names of the event list.

    Parameters
    ----------
    counts_map : `~gammapy.maps.Map`
        Map object, will be filled by this function.
    events : `~gammapy.data.EventList`
        Event list

    Examples
    --------
    To make a counts map, create an empty map with a geometry of your choice
    and then fill it using this function::

        from gammapy.maps import Map
        from gammapy.data import EventList
        from gammapy.cube import fill_map_counts
        events = EventList.read('$GAMMAPY_EXTRA/datasets/cta-1dc/data/baseline/gps/gps_baseline_110380.fits')
        counts = Map.create(coordsys='GAL', skydir=(0, 0), binsz=0.1, npix=(120, 100))
        fill_map_counts(counts, events)
        counts.plot()

    If you have a given map already, and want to make a counts image
    with the same geometry (not using the pixel data from the original map), do this:

        from gammapy.maps import Map
        from gammapy.data import EventList
        from gammapy.cube import fill_map_counts
        events = EventList.read('$GAMMAPY_EXTRA/datasets/fermi_2fhl/2fhl_events.fits.gz')
        reference_map = Map.read('$GAMMAPY_EXTRA/datasets/fermi_2fhl/fermi_2fhl_gc.fits.gz')
        counts = Map.from_geom(reference_map.geom)
        fill_map_counts(counts, events)
        counts.smooth(3).plot()

    It works for IACT and Fermi-LAT events, for WCS or HEALPix map geometries,
    and also for extra axes. Especially energy axes are automatically handled correctly.
    """
    # Make a coordinate dictionary; skycoord is always added
    coord_dict = dict(skycoord=events.radec)

    # Now add one coordinate for each extra map axis
    cols = {k.upper(): v for k, v in events.table.columns.items()}

    for axis in counts_map.geom.axes:
        if axis.name.lower() in ['energy', 'energy_reco']:
            # This axis is the energy. We treat it differently because axis.name could be e.g. 'energy_reco'
            coord_dict[axis.name] = events.energy.to(axis.unit)
        # TODO: add proper extraction for time
        else:
            # We look for other axes name in the table column names (case insensitive)
            try:
                coord_dict[axis.name] = Quantity(cols[axis.name.upper()]).to(axis.unit)
            except KeyError:
                raise KeyError("Column not found in event list: {!r}".format(axis.name))

    counts_map.fill_by_coord(coord_dict)

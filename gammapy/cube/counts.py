# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.units import Quantity

__all__ = ["fill_map_counts"]


def fill_map_counts(counts_map, events):
    """Fill events into a counts map.

    This method handles sky coordinates automatically.
    For all other map axes, an event list table column with the same name
    has to be present (case-insensitive compare), or a KeyError will be raised.

    Note that this function is just a thin wrapper around the sky map
    ``fill_by_coord`` method. For more complex scenarios, use that directly.

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
        events = EventList.read('$GAMMAPY_DATA/cta-1dc/data/baseline/gps/gps_baseline_110380.fits')
        counts = Map.create(coordsys='GAL', skydir=(0, 0), binsz=0.1, npix=(120, 100))
        fill_map_counts(counts, events)
        counts.plot()

    If you have a given map already, and want to make a counts image
    with the same geometry (not using the pixel data from the original map), do this:

        from gammapy.maps import Map
        from gammapy.data import EventList
        from gammapy.cube import fill_map_counts
        events = EventList.read('$GAMMAPY_DATA/fermi_2fhl/2fhl_events.fits.gz')
        reference_map = Map.read('$GAMMAPY_DATA/fermi_2fhl/fermi_2fhl_gc.fits.gz')
        counts = Map.from_geom(reference_map.geom)
        fill_map_counts(counts, events)
        counts.smooth(3).plot()

    It works for IACT and Fermi-LAT events, for WCS or HEALPix map geometries,
    and also for extra axes. Especially energy axes are automatically handled correctly.
    """
    coord = dict(skycoord=events.radec)

    cols = {k.upper(): v for k, v in events.table.columns.items()}

    for axis in counts_map.geom.axes:
        try:
            col = cols[axis.name.upper()]
            coord[axis.name] = Quantity(col).to(axis.unit)
        except KeyError:
            raise KeyError("Column not found in event list: {!r}".format(axis.name))

    counts_map.fill_by_coord(coord)

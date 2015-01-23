# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Event list: table of LON, LAT, ENERGY, TIME
"""
from __future__ import print_function, division
import numpy as np
import logging
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table
from ..data import GoodTimeIntervals, TelescopeArray

__all__ = ['EventList',
           'check_event_list_coordinates',
           ]

logger = logging.getLogger(__name__)


class EventList(object):
    """Event list.

    Table of event parameters and optional extra info.

    TODO: better to split out the events table container
    into a separate class?

    Parameters
    ----------
    events : `~astropy.table.Table`
        Events table
    telescope_array : `~gammapy.data.TelescopeArray`
        Telescope array info
    good_time_intervals : `~gammapy.data.TelescopeArray`
    """
    def __init__(self, events,
                 telescope_array=None,
                 good_time_intervals=None):
        self.events = events
        self.telescope_array = telescope_array
        self.good_time_intervals = good_time_intervals

    @staticmethod
    def from_hdu_list(hdu_list):
        """Create `EventList` from a `~astropy.io.fits.HDUList`.
        """
        # TODO: This doesn't work because FITS / Table is not integrated.
        # Maybe the easiest solution for now it to write the hdu_list
        # to an in-memory buffer with StringIO and then read it
        # back using Table.read()?
        raise NotImplementedError
        events = Table(hdu_list['EVENTS'])
        telescope_array = TelescopeArray.from_hdu(hdu_list['TELARRAY'])
        good_time_intervals = GoodTimeIntervals.from_hdu(hdu_list['GTI'])

        return EventList(events, telescope_array, good_time_intervals)

    @staticmethod
    def read(filename):
        """Read event list from FITS file.
        """
        # return EventList.from_hdu_list(fits.open(filename))
        events = Table.read(filename, hdu='EVENTS')
        telescope_array = Table.read(filename, hdu='TELARRAY')
        good_time_intervals = Table.read(filename, hdu='GTI')

        return EventList(events, telescope_array, good_time_intervals)

    def __str__(self):
        # TODO: implement useful info (min, max, sum)
        s = 'Event list information:'
        s += '- events: {}\n'.format(len(self.events))
        s += '- telescopes: {}\n'.format(len(self.telescope_array))
        s += '- good time intervals: {}\n'.format(len(self.good_time_intervals))
        return s


def _check_event_list_coordinates_galactic(event_list, accuracy):
    """Check if RA / DEC matches GLON / GLAT."""
    events = event_list.events

    for colname in ['RA', 'DEC', 'GLON', 'GLAT']:
        if colname not in events.colnames:
            # GLON / GLAT columns are optional ...
            # so it's OK if they are not present ... just move on ...
            logger.info('Skipping Galactic coordinate check. '
                        'Missing column: "{}".'.format(colname))
            return True

    ra = events['RA'].astype('f64')
    dec = events['DEC'].astype('f64')
    radec = SkyCoord(ra, dec, unit='deg', frame='icrs')

    glon = events['GLON'].astype('f64')
    glat = events['GLAT'].astype('f64')
    galactic = SkyCoord(glon, glat, unit='deg', frame='galactic')

    separation = radec.separation(galactic).to('arcsec')
    max_separation = np.max(separation)

    if max_separation > accuracy:
        # TODO: probably we need to print run number and / or other
        # things for this to be useful in a pipeline ...
        logger.warning('RA/DEC not consistent with GLON/GLAT.'
                       'Max separation: {}'.format(max_separation))
        return False
    else:
        return True


def _check_event_list_coordinates_horizon(event_list, accuracy):
    """Check if ALT / AZ and DETX / DETY matches RA / DEC."""
    events = event_list.events
    meta = event_list.telarray.meta

    location = event_list.telarray.get_earth_location()

    # import IPython; IPython.embed(); 1/0

    return True


def check_event_list_coordinates(event_list, accuracy=Angle('1 arcsec')):
    """Check if various event list coordinates are consistent.

    This can be useful to discover issue in the coordinate
    transformations and FITS exporters of each experiment
    (HESS, VERITAS, CTA, ...)

    This is a chatty function that emits log messages.

    Parameters
    ----------
    event_list : `EventList`
        Event list
    accuracy : `~astropy.coordinates.Angle`
        Required accuracy.

    Returns
    -------
    status : bool
        All coordinates consistent?
    """
    ok = True
    ok &= _check_event_list_coordinates_galactic(event_list, accuracy)
    ok &= _check_event_list_coordinates_horizon(event_list, accuracy)
    return ok

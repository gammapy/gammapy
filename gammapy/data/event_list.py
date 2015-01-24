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

__all__ = ['EventListDataset',
           'EventList',
           'check_event_list_coordinates',
           ]

logger = logging.getLogger(__name__)


class EventList(Table):
    """Event list table.

     Reconstructed event parameters:
     - Time
     - Position
     - Energy
     - ...

    TODO: should this be private or public?
    I.e. does the end-user ever need to use it or is
    interacting with `EventDataset` enough?
    """
    pass


class EventListDataset(object):
    """Event list dataset (event list plus some extra info).

    TODO: I'm not sure if IRFs should be included in this
    class or if an extra container class should be added.

    Parameters
    ----------
    event_list : `~gammapy.data.EventList`
        Event list table
    telescope_array : `~gammapy.data.TelescopeArray`
        Telescope array info
    good_time_intervals : `~gammapy.data.GoodTimeIntervals`
        Observation time interval info
    """
    def __init__(self, event_list,
                 telescope_array=None,
                 good_time_intervals=None):
        self.event_list = event_list
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
        event_list = EventList.from_hdu(hdu_list['EVENTS'])
        telescope_array = TelescopeArray.from_hdu(hdu_list['TELARRAY'])
        good_time_intervals = GoodTimeIntervals.from_hdu(hdu_list['GTI'])

        return EventListDataset(event_list, telescope_array, good_time_intervals)

    @staticmethod
    def read(filename):
        """Read event list from FITS file.
        """
        # return EventList.from_hdu_list(fits.open(filename))
        event_list = EventList.read(filename, hdu='EVENTS')
        telescope_array = TelescopeArray.read(filename, hdu='TELARRAY')
        good_time_intervals = GoodTimeIntervals.read(filename, hdu='GTI')

        return EventListDataset(event_list, telescope_array, good_time_intervals)

    def __str__(self):
        # TODO: implement useful info (min, max, sum)
        s = 'Event list dataset information:'
        s += '- events: {}\n'.format(len(self.events))
        s += '- telescopes: {}\n'.format(len(self.telescope_array))
        s += '- good time intervals: {}\n'.format(len(self.good_time_intervals))
        return s


def _check_event_list_coordinates_galactic(event_list_dataset, accuracy):
    """Check if RA / DEC matches GLON / GLAT."""
    event_list = event_list_dataset.event_list

    for colname in ['RA', 'DEC', 'GLON', 'GLAT']:
        if colname not in event_list.colnames:
            # GLON / GLAT columns are optional ...
            # so it's OK if they are not present ... just move on ...
            logger.info('Skipping Galactic coordinate check. '
                        'Missing column: "{}".'.format(colname))
            return True

    ra = event_list['RA'].astype('f64')
    dec = event_list['DEC'].astype('f64')
    radec = SkyCoord(ra, dec, unit='deg', frame='icrs')

    glon = event_list['GLON'].astype('f64')
    glat = event_list['GLAT'].astype('f64')
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


def _check_event_list_coordinates_horizon(event_list_dataset, accuracy):
    """Check if ALT / AZ matches RA / DEC."""
    event_list = event_list_dataset.event_list
    location = event_list.telescope_array.get_earth_location()

    import IPython; IPython.embed(); 1/0
    # TODO: convert RA / DEC to ALT / AZ and then compute
    # separation in ALT / AZ

    return True


def _check_event_list_coordinates_field_of_view(event_list_dataset, accuracy):
    """Check if DETX / DETY matches ALT / AZ"""
    return True


def check_event_list_coordinates(event_list_dataset, accuracy=Angle('1 arcsec')):
    """Check if various event list coordinates are consistent.

    This can be useful to discover issue in the coordinate
    transformations and FITS exporters of each experiment
    (HESS, VERITAS, CTA, ...)

    This is a chatty function that emits log messages.

    Parameters
    ----------
    event_list : `~gammapy.data.EventListDataset`
        Event list
    accuracy : `~astropy.coordinates.Angle`
        Required accuracy.

    Returns
    -------
    status : bool
        All coordinates consistent?
    """
    ok = True
    ok &= _check_event_list_coordinates_galactic(event_list_dataset, accuracy)
    ok &= _check_event_list_coordinates_horizon(event_list_dataset, accuracy)
    ok &= _check_event_list_coordinates_field_of_view(event_list_dataset, accuracy)
    return ok

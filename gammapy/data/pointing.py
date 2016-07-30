# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.utils import lazyproperty
from astropy.units import Quantity
from astropy.table import Table
from astropy.coordinates import SkyCoord, AltAz
from ..utils.scripts import make_path
from ..time.utils import time_ref_from_dict
from .utils import _earth_location_from_dict

import numpy as np

__all__ = [
    'PointingInfo',
]


class PointingInfo(object):
    """IACT array pointing info.

    TODO: link to open spec.
    TODO: share code with the `~gammapy.data.EventList` and `~gammapy.data.ObservationTable` classes.

    This class has many cached properties.
    Should be used as read-only.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table (with meta header info) on pointing

    Examples
    --------

    >>> from gammapy.data import PointingInfo
    >>> pointing_info = PointingInfo.read('$GAMMAPY_EXTRA/test_datasets/hess_event_list.fits')
    >>> print(pointing_info)
    """

    def __init__(self, table):
        self.table = table

    @classmethod
    def read(cls, filename, hdu=None):
        """Read `PointingInfo` table from file.

        Parameters
        ----------
        filename : str
            File name
        hdu : int or str
            HDU number or name

        Returns
        -------
        pointing_info : `PointingInfo`
            Pointing info object
        """
        filename = make_path(filename)

        if hdu is None:
            hdu = 'POINTING'

        table = Table.read(str(filename), hdu=hdu)
        return cls(table=table)

    def __str__(self):
        """Basic info."""
        ss = 'Pointing info:\n\n'
        ss += 'Location:     {}\n'.format(self.location.geodetic)
        m = self.table.meta
        ss += 'MJDREFI, MJDREFF, TIMESYS = {}\n'.format((m['MJDREFI'], m['MJDREFF'], m['TIMESYS']))
        ss += 'Time ref:     {}\n'.format(self.time_ref.fits)
        ss += 'Time ref:     {} MJD (TT)\n'.format(self.time_ref.mjd)
        sec = self.duration.to('second').value
        hour = self.duration.to('hour').value
        ss += 'Duration:     {} sec = {} hours\n'.format(sec, hour)
        ss += 'Table length: {}\n'.format(len(self.table))

        ss += '\nSTART:\n' + self._str_for_index(0) + '\n'
        ss += '\nEND:\n' + self._str_for_index(-1) + '\n'

        return ss

    def _str_for_index(self, idx):
        """Information for one point in the pointing table"""
        ss = 'Time:  {}\n'.format(self.time[idx].fits)
        ss += 'Time:  {} MJD (TT)\n'.format(self.time[idx].mjd)
        ss += 'RADEC: {} deg\n'.format(self.radec[idx].to_string())
        ss += 'ALTAZ: {} deg\n'.format(self.altaz[idx].to_string())
        return ss

    def _str_for_time(self, t):
        """Information for an arbitrary time"""

        if t < self.time[0]:
            return 'Pointing at time {} was requested but history begins at {}'.format(t.mjd, self.time[0].mjd)

        if t > self.time[-1]:
            return 'Pointing at time {} was requested but history ends at {}'.format(t.mjd, self.time[-1].mjd)

        times_before = np.where(self.time < t)
        idx_before = times_before[0][-1]

        times_after = np.where(self.time > t)
        idx_after = times_after[0][0]

        alt_before = self.altaz[idx_before].alt
        az_before = self.altaz[idx_before].az

        alt_after = self.altaz[idx_after].alt
        az_after = self.altaz[idx_after].az

        alt = 0.5 * alt_before.value + 0.5 * alt_after.value

        az = 0.5 * az_before.value + 0.5 * az_after.value

        if abs(az_after.value - az_before.value) > 180:
            az += 180

        ra_before = self.radec[idx_before].ra
        dec_before = self.radec[idx_before].dec

        ra_after = self.radec[idx_after].ra
        dec_after = self.radec[idx_after].dec

        ra = 0.5 * ra_before.value + 0.5 * ra_after.value

        dec = 0.5 * dec_before.value + 0.5 * dec_after.value

        if abs(ra_after.value - ra_before.value) > 180:
            ra += 180

        if abs(dec_after.value - dec_before.value) > 180:
            dec += 180

        ss = 'Time:  {} MJD (TT)\n'.format(t.mjd)
        ss += 'RADEC: {} {} deg\n'.format(ra, dec)
        ss += 'ALTAZ: {} {} deg\n'.format(az, alt)
        return ss

    @lazyproperty
    def location(self):
        """Observatory location (`~astropy.coordinates.EarthLocation`)"""
        return _earth_location_from_dict(self.table.meta)

    @lazyproperty
    def time_ref(self):
        """Time reference (`~astropy.time.Time`)"""
        # For debugging ... change TIMESYS
        # self.table.meta['TIMESYS'] = 'utc'
        return time_ref_from_dict(self.table.meta)

    @lazyproperty
    def duration(self):
        """Pointing table duration (`~astropy.time.TimeDelta`).

        The time difference between the first and last entry.
        """
        return self.time[-1] - self.time[0]

    @lazyproperty
    def time(self):
        """Time array (`~astropy.time.Time`)"""
        met = Quantity(self.table['TIME'].astype('float64'), 'second')
        time = self.time_ref + met
        return time.tt

    @lazyproperty
    def radec(self):
        """RA / DEC position from table (`~astropy.coordinates.SkyCoord`)"""
        lon = self.table['RA_PNT'].astype('float64')
        lat = self.table['DEC_PNT'].astype('float64')
        return SkyCoord(lon, lat, unit='deg', frame='icrs')

    @lazyproperty
    def altaz_frame(self):
        """ALT / AZ frame (`~astropy.coordinates.AltAz`)."""
        return AltAz(obstime=self.time, location=self.location)

    @lazyproperty
    def altaz(self):
        """ALT / AZ position from table (`~astropy.coordinates.SkyCoord`)"""
        lon = self.table['AZ_PNT'].astype('float64')
        lat = self.table['ALT_PNT'].astype('float64')
        return SkyCoord(lon, lat, unit='deg', frame=self.altaz_frame)

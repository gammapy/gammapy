# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.utils import lazyproperty
from astropy.units import Quantity
from astropy.table import Table
from astropy.coordinates import SkyCoord, AltAz
from ..utils.scripts import make_path
from ..utils.time import time_ref_from_dict
from .utils import _earth_location_from_dict

__all__ = [
    'PointingInfo',
]


# TODO: share code with the `~gammapy.data.EventList` and `~gammapy.data.ObservationTable` classes.
class PointingInfo(object):
    """IACT array pointing info.

    Data format specification: :ref:`gadf:iact-pnt`

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
        """Information for one point in the pointing table."""
        ss = 'Time:  {}\n'.format(self.time[idx].fits)
        ss += 'Time:  {} MJD (TT)\n'.format(self.time[idx].mjd)
        ss += 'RADEC: {} deg\n'.format(self.radec[idx].to_string())
        ss += 'ALTAZ: {} deg\n'.format(self.altaz[idx].to_string())
        return ss

    @lazyproperty
    def location(self):
        """Observatory location (`~astropy.coordinates.EarthLocation`)."""
        return _earth_location_from_dict(self.table.meta)

    @lazyproperty
    def time_ref(self):
        """Time reference (`~astropy.time.Time`)"""
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

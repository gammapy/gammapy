# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.units import Quantity
from astropy.table import Table
from ..utils.time import time_ref_from_dict
from ..utils.scripts import make_path

__all__ = [
    'GTI',
]


class GTI(object):
    """Good time intervals (GTI) `~astropy.table.Table`.

    Data format specification: :ref:`gadf:iact-gti`

    Note: at the moment dead-time and live-time is in the
    EVENTS header ... the GTI header just deals with
    observation times.

    Parameters
    ----------
    table : `~astropy.table.Table`
        GTI table

    Examples
    --------
    Load GTIs for a H.E.S.S. event list:

    >>> from gammapy.data import GTI
    >>> filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz'
    >>> gti = GTI.read(filename)
    >>> print(gti)
    GTI info:
    - Number of GTIs: 1
    - Duration: 1568.0 s
    - Start: 53292.00592592593 MET
    - Start: 2004-10-14T00:08:32.000(TT)
    - Stop: 53292.02407407408 MET
    - Stop: 2004-10-14T00:34:40.000(TT)

    Load GTIs for a Fermi-LAT event list:

    >>> filename = '$GAMMAPY_EXTRA/datasets/fermi_2fhl/2fhl_events.fits.gz'
    >>> gti = GTI.read(filename)
    >>> print(gti)
    GTI info:
    - Number of GTIs: 36589
    - Duration: 171273490.97510204 s
    - Start: 54682.659499814814 MET
    - Start: 2008-08-04T15:49:40.784(TT)
    - Stop: 57053.993550740735 MET
    - Stop: 2015-01-31T23:50:42.784(TT)
    """

    def __init__(self, table):
        self.table = table

    @classmethod
    def read(cls, filename, **kwargs):
        """Read from FITS file.

        Parameters
        ----------
        filename : `~gammapy.extern.pathlib.Path`, str
            Filename
        """
        filename = make_path(filename)
        if 'hdu' not in kwargs:
            kwargs.update(hdu='GTI')
        table = Table.read(str(filename), **kwargs)
        return cls(table=table)

    def __str__(self):
        ss = 'GTI info:\n'
        ss += '- Number of GTIs: {}\n'.format(len(self.table))
        ss += '- Duration: {}\n'.format(self.time_sum)
        ss += '- Start: {} MET\n'.format(self.time_start[0])
        ss += '- Start: {}\n'.format(self.time_start[0].fits)
        ss += '- Stop: {} MET\n'.format(self.time_stop[-1])
        ss += '- Stop: {}\n'.format(self.time_stop[-1].fits)
        return ss

    @property
    def time_delta(self):
        """GTI durations in seconds (`~astropy.units.Quantity`)."""
        start = self.table['START'].astype('float64')
        stop = self.table['STOP'].astype('float64')
        return Quantity(stop - start, 'second')

    @property
    def time_sum(self):
        """Sum of GTIs in seconds (`~astropy.units.Quantity`)."""
        return self.time_delta.sum()

    @property
    def time_start(self):
        """GTI start times (`~astropy.time.Time`)."""
        met_ref = time_ref_from_dict(self.table.meta)
        met = Quantity(self.table['START'].astype('float64'), 'second')
        return met_ref + met

    @property
    def time_stop(self):
        """GTI end times (`~astropy.time.Time`)."""
        met_ref = time_ref_from_dict(self.table.meta)
        met = Quantity(self.table['STOP'].astype('float64'), 'second')
        return met_ref + met

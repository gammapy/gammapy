# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
from astropy.units import Quantity
from astropy.table import Table
from ..time import time_ref_from_dict
from ..utils.scripts import make_path

__all__ = [
    'GTI',
]


class GTI(Table):
    """Good time intervals (GTI) `~astropy.table.Table`.

    Note: at the moment dead-time and live-time is in the
    EVENTS header ... the GTI header just deals with
    observation times.
    """

    def __init__(self, *args, **kwargs):
        super(GTI, self).__init__(*args, **kwargs)

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
        return super(GTI, cls).read(str(filename), **kwargs)

    def summary(self, file=None):
        """Summary info string."""
        if not file:
            file = sys.stdout

        print('GTI info:', file=file)
        print('- Number of GTIs: {}'.format(len(self)), file=file)
        print('- Duration: {}'.format(self.time_sum), file=file)
        print('- Start: {}'.format(self.time_start[0]), file=file)
        print('- Stop: {}'.format(self.time_stop[-1]), file=file)

    @property
    def time_delta(self):
        """GTI durations in seconds (`~astropy.units.Quantity`)."""
        start = self['START'].astype('float64')
        stop = self['STOP'].astype('float64')
        return Quantity(stop - start, 'second')

    @property
    def time_sum(self):
        """Sum of GTIs in seconds (`~astropy.units.Quantity`)."""
        return self.time_delta.sum()

    @property
    def time_start(self):
        """GTI start times (`~astropy.time.Time`)."""
        met_ref = time_ref_from_dict(self.meta)
        met = Quantity(self['START'].astype('float64'), 'second')
        return met_ref + met

    @property
    def time_stop(self):
        """GTI end times (`~astropy.time.Time`)."""
        met_ref = time_ref_from_dict(self.meta)
        met = Quantity(self['STOP'].astype('float64'), 'second')
        return met_ref + met

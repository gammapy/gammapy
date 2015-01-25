# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.units import Quantity
from astropy.table import Table
from . import utils

__all__ = ['GoodTimeIntervals']


class GoodTimeIntervals(Table):
    """Good time intervals (GTI) `~astropy.table.Table`.

    Note: at the moment dead-time and live-time is in the
    EVENTS header ... the GTI header just deals with
    observation times.
    """
    def __init__(self, *args, **kwargs):
        super(GoodTimeIntervals, self).__init__(*args, **kwargs)

    @property
    def info(self):
        """Summary info string."""
        s = '---> Good time interval (GTI) info:\n'
        s += '- Number of GTIs: {}\n'.format(len(self))
        s += '- Duration: {}\n'.format(self.time_sum)
        s += '- Start: {}\n'.format(self.time_start[0])
        s += '- Stop: {}\n'.format(self.time_stop[-1])
        return s

    @property
    def time_delta(self):
        """GTI durations in seconds (`~astropy.units.Quantity`)."""
        start = self['START'].astype('f64')
        stop = self['STOP'].astype('f64')
        return Quantity(stop - start, 'second')

    @property
    def time_sum(self):
        """Sum of GTIs in seconds (`~astropy.units.Quantity`)."""
        return self.time_delta.sum()

    @property
    def time_start(self):
        """GTI start times (`~astropy.time.Time`)."""
        met_ref = utils._time_ref_from_dict(self.meta)
        met = Quantity(self['START'].astype('f64'), 'second')
        return met_ref + met

    @property
    def time_stop(self):
        """GTI end times (`~astropy.time.Time`)."""
        met_ref = utils._time_ref_from_dict(self.meta)
        met = Quantity(self['STOP'].astype('f64'), 'second')
        return met_ref + met

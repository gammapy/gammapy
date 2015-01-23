# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from astropy.table import Table
from astropy.units import Quantity

__all__ = ['GoodTimeIntervals']


class GoodTimeIntervals(Table):
    """Good time intervals (GTI) container.
    """

    def __str__(self):
        # TODO: implement useful info (min, max, sum)
        return str(self)

    def gti_lengths(self):
        """List of GTI lengths."""
        table = self
        times = table['TSTOP'] - table['TSTART']
        return Quantity(times, 'second')

    def total_time(self):
        """Sum of GTIs."""
        return np.sum(self.gti_intervals)

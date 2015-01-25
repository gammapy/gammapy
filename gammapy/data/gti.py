# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging
import numpy as np
from astropy.table import Table
from astropy.units import Quantity
from ..data import InvalidDataError

__all__ = ['GoodTimeIntervals']

logger = logging.getLogger(__name__)



class GoodTimeIntervals(Table):
    """Good time intervals (GTI) `~astropy.table.Table`.
    """

    # TODO: return absolute times as `astropy.time.Time` objects using
    # MJDREFI and MJDREFF from the header.
    # TODO: return time intervals as `astropy.time.TimeDelta` objects?

    def __init__(self, *args, **kwargs):
        super(GoodTimeIntervals, self).__init__(*args, **kwargs)

        self._checks()
        self._init()

    def _init(self):
        """Initialize some data members."""
        pass

    def _checks(self):
        """Check some invariants that should always hold."""
        # Check that required info is there
        for colname in ['START', 'STOP']:
            if colname not in self.colnames:
                raise InvalidDataError('GTI missing column: {}'.format(colname))

        for key in ['TSTART', 'TSTOP', 'MJDREFI', 'MJDREFF']:
            if key not in self.meta:
                raise InvalidDataError('GTI missing header keyword: {}'.format(key))

        # TODO: Check that header keywords agree with table entries
        # TSTART, TSTOP, MJDREFI, MJDREFF

        # Check that START and STOP times are consecutive
        times = np.ravel(self['START'], self['STOP'])
        # TODO: not sure this is correct ... add test with a multi-gti table from Fermi.
        if not np.all(np.diff(times) >= 0):
            raise InvalidDataError('GTIs are not consecutive or sorted.')

    @property
    def info(self):
        """Summary info string."""
        s = '---> Good time interval info:\n'
        s += '- Number of intervals: {}\n'.format(len(self))
        s += '- Start: {}\n'.format(self.time_start)
        s += '- Stop: {}\n'.format(self.time_stop)
        s += '- Observation: {}\n'.format(self.time_observation)
        s += '- Live: {}\n'.format(self.time_live)
        s += '- Deadtime fraction: {}\n'.format(self.time_dead_fraction)
        return s

    @property
    def time_observations(self):
        """List of GTI durations (`~astropy.time.TimeDelta`)."""
        times = self['STOP'] - self['START']
        return Quantity(times, 'second')

    @property
    def time_observation(self):
        """Sum of GTIs (`~astropy.time.TimeDelta`)."""
        return self.time_observations.sum()

    @property
    def time_start(self):
        """Start time of first GTI (`~astropy.time.Time`)."""
        return Quantity(self['START'][0], 'second')

    @property
    def time_stop(self):
        """End time of last GTI (`~astropy.time.Time`)."""
        return Quantity(self['STOP'][-1], 'second')

    @property
    def time_dead_fraction(self):
        """Dead-time fraction.

        Defined as dead-time over observation time.

        Dead-time is defined as the time during the observation
        where the detector didn't record events:
        http://en.wikipedia.org/wiki/Dead_time
        http://adsabs.harvard.edu/abs/2004APh....22..285F

        The dead-time fraction is used in the live-time computation,
        which in turn is used in the exposure and flux computation.
        """
        logger.warning('Assuming zero dead-time. '
                       '(At the moment deadtime info is only stored in the EVENTS header!?')
        return 0

    @property
    def time_live(self):
        """Live-time (`~astropy.time.TimeDelta`).

        The dead-time-corrected observation time.

        Computed as ``t_live = t_observation * (1 - f_dead)``
        where ``f_dead`` is the dead-time fraction.
        """
        return self.time_observation * (1 - self.time_dead_fraction)

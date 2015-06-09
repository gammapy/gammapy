# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.table import Table

__all__ = [
    # 'Observation',
    'ObservationTable',
]


class Observation(object):
    """Observation (a.k.a. run).

    TODO: not clear if this class is useful.

    Parameters
    ----------
    TODO
    """
    def __init__(self, GLON, GLAT, livetime=1800,
                 eff_area=1e12, background=0):
        self.GLON = GLON
        self.GLAT = GLAT
        self.livetime = livetime

    def wcs_header(self, system='FOV'):
        """Create a WCS FITS header for an per-run image.

        The image is centered on the run position in one of these systems:
        FOV, Galactic, Equatorial
        """
        raise NotImplementedError


class ObservationTable(Table):
    """Observation table (a.k.a. run list).

    This is an `~astropy.table.Table` sub-class, with a few
    convenience methods and the following columns:

    * ``OBS_ID``
    * ``TIME_OBSERVATION``
    * ``TIME_LIVE``
    * ``TIME_START``
    * ``TIME_STOP``
    * ``AZ``
    * ``ALT``
    * ``RA``
    * ``DEC``
    * ...

    """

    def info(self):
        ss = 'Observation table:\n'
        ss += 'Number of observations: {}\n'.format(len(self))
        ontime = self['TIME_OBSERVATION'].sum()
        ss += 'Total observation time: {}\n'.format(ontime)
        livetime = self['TIME_LIVE'].sum()
        ss += 'Total live time: {}\n'.format(livetime)
        dtf = 100. * (1 - livetime / ontime)
        ss += 'Average dead time fraction: {:5.2f}%'.format(dtf)
        #TODO: units are not shown!!!
        return ss

    def select_linspace_subset(self, num):
        """Select subset of observations.

        This is mostly useful for testing, if you want to make
        the analysis run faster.

        TODO: implement more methods to subset and split observation lists
        as well as functions to summarise differences between
        observation lists and e.g. select the common subset.

        Parameters
        ----------
        num : int
            Number of samples to select.

        Returns
        -------
        table : `ObservationTable`
            Subset observation table (a copy).
        """
        indices = np.linspace(start=0, stop=len(self), num=num, endpoint=False)
        # Round down to nearest integer
        indices = indices.astype('int')
        return self[indices]


# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Run and RunList class"""
from __future__ import print_function, division

__all__ = ['Run', 'RunList']


class Run(object):
    """Run parameters container.

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


class RunList(list):
    """Run list container.
    """

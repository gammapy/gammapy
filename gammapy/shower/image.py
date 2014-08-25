# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from ..shower import hillas_parameters

__all__ = ['ShowerImage']


class ShowerImage(object):
    """Air shower image.

    TODO: implement.
    - neighbor list iterator
    - I/O
    - unit test
    - plotting
    - Gauss model fit
    """
    def __init__(self, x, y, s):
        self.x = x
        self.y = y
        self.s = s

    def hillas_parameters(self):
        """Compute Hillas parameters."""
        return hillas_parameters(self.x, self.y, self.s)

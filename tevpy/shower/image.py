# Licensed under a 3-clause BSD style license - see LICENSE.rst
from .hillas import hillas_parameters

__all__ = ['ShowerImage']

class ShowerImage(object):
    """Air shower image.
    
    TODO: implement.
    """
    def __init__(self, x, y, s):
        self.x = x
        self.y = y
        self.s = s
    
    def hillas_parameters(self):
        """Compute Hillas parameters.
        """
        return hillas_parameters(self.x, self.y, self.s)

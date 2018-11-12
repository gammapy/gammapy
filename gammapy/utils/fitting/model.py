# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import copy

__all__ = ["Model"]


class Model(object):
    """Model base class."""

    # TODO: expose model parameters as attributes

    def copy(self):
        """A deep copy."""
        return copy.deepcopy(self)

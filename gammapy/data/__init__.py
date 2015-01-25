# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Data classes.
"""


class InvalidDataError(Exception):
    """Invalid data found."""


class CountsCubeDataset(object):
    """Counts cube dataset."""
    # TODO: implement me


class CountsSpectrumDataset(object):
    """Counts spectrum dataset."""
    # TODO: implement me


class CountsImageDataset(object):
    """Counts image dataset."""
    # TODO: implement me


class CountsLightCurveDataset(object):
    """Counts light-curve dataset."""
    # TODO: implement me


from .gti import *
from .telarray import *
from .event_list import *
from .spectral_cube import *

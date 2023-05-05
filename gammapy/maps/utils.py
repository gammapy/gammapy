# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle


def _check_width(width):
    """Check and normalise width argument.

    Always returns tuple (lon, lat) as float in degrees.
    """
    if isinstance(width, tuple):
        lon = Angle(width[0], "deg").deg
        lat = Angle(width[1], "deg").deg
        return lon, lat
    else:
        angle = Angle(width, "deg").deg
        if np.isscalar(angle):
            return angle, angle
        else:
            return tuple(angle)


def _check_binsz(binsz):
    """Check and normalise bin size argument.

    Always returns an object with the same shape
    as the input where the spatial coordinates
    are a float in degrees.
    """
    if isinstance(binsz, tuple):
        lon_sz = Angle(binsz[0], "deg").deg
        lat_sz = Angle(binsz[1], "deg").deg
        return lon_sz, lat_sz
    elif isinstance(binsz, list):
        binsz[:2] = Angle(binsz[:2], unit="deg").deg
        return binsz
    return Angle(binsz, unit="deg").deg


def coordsys_to_frame(coordsys):
    if coordsys in ["CEL", "C"]:
        return "icrs"
    elif coordsys in ["GAL", "G"]:
        return "galactic"
    else:
        raise ValueError(f"Unknown coordinate system: '{coordsys}'")


def frame_to_coordsys(frame):
    if frame in ["icrs", "fk5", "fk4"]:
        return "CEL"
    elif frame in ["galactic"]:
        return "GAL"
    else:
        raise ValueError(f"Unknown coordinate frame '{frame}'")


class InvalidValue:
    """Class to define placeholder for invalid array values."""

    float = np.nan
    int = np.nan
    bool = np.nan

    def __getitem__(self, dtype):
        if np.issubdtype(dtype, np.integer):
            return self.int
        elif np.issubdtype(dtype, np.floating):
            return self.float
        elif np.issubdtype(dtype, np.dtype(bool).type):
            return self.bool
        else:
            raise ValueError(f"No invalid value placeholder defined for {dtype}")


class InvalidIndex:
    """Class to define placeholder for invalid array indices."""

    float = np.nan
    int = -1
    bool = False


INVALID_VALUE = InvalidValue()
INVALID_INDEX = InvalidIndex()


def edges_from_lo_hi(edges_lo, edges_hi):
    if np.isscalar(edges_lo.value) and np.isscalar(edges_hi.value):
        return u.Quantity([edges_lo, edges_hi])

    edges = edges_lo.copy()
    try:
        edges = edges.insert(len(edges), edges_hi[-1])
    except AttributeError:
        edges = np.insert(edges, len(edges), edges_hi[-1])
    return edges

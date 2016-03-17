"""Coordinate transformation methods using PyEphem

http://rhodesmill.org/pyephem/
"""

import numpy as np
import ephem
from kapteyn.wcs import Projection

def approximate_nominal_to_altaz(nominal, horizon_center=(0, 0)):
    """Transform nominal coordinates to horizon coordinates.

    nominal = (x, y) in meter
    horizon_center = (az_center, alt_center) in deg

    Returns: horizon = (az, alt) in deg

    TODO: The following method of computing Alt / Az is only
    an approximation. Implement and use a utility function
    using the TAN FITS projection.
    """
    x, y = np.asarray(nominal, dtype='float64')
    az_center, alt_center = np.asarray(horizon_center, dtype='float64')

    # Note: alt increases where x increases, az increases where y increases
    az = az_center + np.degrees(np.tan(y)) / np.cos(np.radians(alt_center))
    alt = alt_center + np.degrees(np.tan(x))

    return az, alt


def nominal_to_altaz(nominal, horizon_center=(0, 0)):
    """Transform nominal coordinates to horizon coordinates.

    nominal = (x, y) in meter
    horizon_center = (az_center, alt_center) in deg

    Returns: horizon = (az, alt) in deg
    """
    x, y = np.asarray(nominal, dtype='float64')
    az_center, alt_center = np.asarray(horizon_center, dtype='float64')
    header = {'NAXIS': 2,
              'NAXIS1': 100,
              'NAXIS2': 100,
              'CTYPE1': 'RA---TAN',
              'CRVAL1': az_center,
              'CRPIX1': 0,
              'CUNIT1': 'deg',
              'CDELT1': np.degrees(1),
              'CTYPE2': 'DEC--TAN',
              'CRVAL2': alt_center,
              'CRPIX2': 0,
              'CUNIT2': 'deg',
              'CDELT2': np.degrees(1),
              }
    projection = Projection(header)
    altaz = projection.toworld((y, x))
    return altaz[0], altaz[1]

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Coordinate transformation methods using PyEphem

http://rhodesmill.org/pyephem/
"""

import numpy as np
import ephem
from kapteyn.wcs import Projection


def HESS(date='2010/00/00 00:00', time_offset=0, epoch='2000', pressure=0):
    """Return an ephem.Observer representing the HESS array center position. 

    Parameters
    ----------
    time_offset: added to date (in seconds) e.g. to specify an event time
                 in a run.
    pressure = 0 turns off refraction.
    """
    lon = 16.500222222222224 # 16 + 30 / 60. + 0.8 / 3600.
    #lon_west = 343.49977777777775 # 360. - lon
    lat = -23.27177777777778 # -(23 + 16 / 60. + 18.4 / 3600.)
    height = 1835
    #height_at_00 = 6.37813594897491205e+06
    # Set up an observer
    observer = ephem.Observer()
    observer.lon = np.radians(lon)
    observer.lat = np.radians(lat)
    observer.elevation = height

    observer.date = date
    observer.date += time_offset * ephem.second
    observer.epoch = epoch
    observer.pressure = pressure
    return observer


def transform(in_position, in_system, out_system, observer=None, use360=False):
    """Simple coordinate transforms between commonly used coordinate systems.

    Parameters
    ----------
    position: input position as a float tuple (longitude, latitude) in degrees
    in_system, out_system: coordinate system strings; one of
        horizon:    Alt,  Az
        equatorial: RA,   DEC
        galactic:   GLON, GLAT
    observer: ephem.Observer (needed when converting to / from horizon system

    use360: use longitude range 0 .. 360 if True, else -180 .. +180

    Returns: transformed input position as a float tuple (longitude, latitude) in degrees

    See http://stackoverflow.com/questions/11169523/how-to-compute-alt-az-for-given-galactic-coordinate-glon-glat-with-pyephem
    """
    # Set the default observer
    observer = observer if observer else HESS()
    # Internally use radians;
    in_position = np.radians(in_position)

    # Transform in_position to Equatorial coordinates ra, dec:
    if in_system == 'horizon':
        ra, dec = map(float, observer.radec_of(in_position[0], in_position[1]))
    elif in_system == 'equatorial':
        ra, dec = in_position
    elif in_system == 'galactic':
        galactic = ephem.Galactic(in_position[0], in_position[1])
        equatorial = ephem.Equatorial(galactic)
        ra, dec = equatorial.ra.real, equatorial.dec.real
    else:
        raise RuntimeError('in_system = %s not supported' % in_system)

    # Here we have ra, dec in radians as floats

    # Now transform Equatorial coordinates to out_system:
    if out_system == 'horizon':
        equatorial = ephem.Equatorial(ra, dec)
        body = ephem.FixedBody()
        body._ra = equatorial.ra
        body._dec = equatorial.dec
        body._epoch = equatorial.epoch
        body.compute(observer)
        out_position = body.az, body.alt
    elif out_system == 'equatorial':
        out_position = ra, dec
    elif out_system == 'galactic':
        equatorial = ephem.Equatorial(ra, dec)
        galactic = ephem.Galactic(equatorial)
        out_position = galactic.lon.real, galactic.lat.real
    else:
        raise RuntimeError('out_system = %s not supported' % out_system)

    out_position = np.degrees(out_position)

    x = out_position[0]
    if use360:
        # Clip longitude to 0 .. 360 deg range
        if x > 360:
            x -= 360
        if x < 0:
            x += 360
    else:

        # Clip longitude to -180 .. +180 deg range
        if x > 180:
            x -= 360
        if x < -180:
            x += 360
    out_position[0] = x

    # Return out position in degrees
    return out_position


def _Galactic_position_angle(GLON, GLAT, observer, eps=1e-3):
    """Compute position angle for the zenith direction in Galactic coordinates.

    Parameters
    ----------
    eps : float
        Step in altitude direction in degrees

    Returns
    -------
    Position angle in deg
    """
    # Compute pointing position in horizon system
    az1, alt1 = transform((GLON, GLAT), 'galactic', 'horizon', observer)
    # Take an eps step *up* in alt in the horizon system
    az2, alt2 = az1, alt1 + eps
    # Compute Galactic coordinate of the second point
    GLON2, GLAT2 = transform((az2, alt2), 'horizon', 'galactic', observer)
    # Measure position angle between the two points.
    # @note: There is a cos(GLAT) factor in the dx computation.
    dx, dy = (GLON2 - GLON) / np.cos(np.radians(GLAT)), GLAT2 - GLAT
    angle = np.arctan2(dx, dy)
    return np.degrees(angle)


def compute_position_angle(run, observer=None, time_method='center'):
    """Compute position angle for the zenith direction in Galactic coordinates for one run.

    Parameters
    ----------
    eps : step in zenith direction in degrees
    """
    observer = observer if observer else HESS()

    GLON, GLAT = run['GLON'], run['GLAT']
    Start_Time = run['Start_Time']

    if time_method == 'start':
        # Compute position angle at the start of the run
        observer.date = Start_Time
        return _Galactic_position_angle(GLON, GLAT, observer)
    elif time_method == 'center':
        # Compute position angle in the middle of the run
        Duration = run['Duration']
        observer.date = Start_Time
        observer.date += Duration * ephem.second
        return _Galactic_position_angle(GLON, GLAT, observer)
    elif time_method == 'average':
        # Compute average position vector over the run
        #Duration = run['Duration']
        #n_time_steps = 10
        raise NotImplementedError
    else:
        raise ValueError('Invalid time_method: %s' % time_method)


def compute_position_angles(runs, observer=None, time_method='center'):
    """Compute the position angle of a vector pointing to zenith
    in Galactic coordinates for a given list of runs."""
    observer = observer if observer else HESS()

    position_angle = np.zeros(len(runs), 'f')
    for ii, run in enumerate(runs):
        position_angle[ii] = compute_position_angle(run, observer, time_method)
    return position_angle


def test_HESS():
    observer = HESS()
    print observer


def test_transform():
    """Test all coordinate transform methods for one test example.

    For three systems there are six possible transformations.

    TODO: Rewrite these tests as unit tests
    """
    # Example observer:
    observer = HESS()
    # Example coordinate:
    galactic0 = 0., 0.
    print 'galactic0:', galactic0

    print('\ngalactic <-> equatorial:')
    equatorial1 = transform(galactic0, 'galactic', 'equatorial')
    print 'equatorial1:', equatorial1
    galactic1 = transform(equatorial1, 'equatorial', 'galactic')
    print 'galactic1:', galactic1

    print('\ngalactic <-> horizon:')
    horizon1 = transform(galactic0, 'galactic', 'horizon', observer)
    print 'horizon1:', horizon1
    galactic2 = transform(horizon1, 'horizon', 'galactic', observer)
    print 'galactic2:', galactic2

    print('\nequatorial <-> horizon')
    horizon2 = transform(equatorial1, 'equatorial', 'horizon', observer)
    print 'horizon2:', horizon2
    equatorial2 = transform(horizon2, 'horizon', 'equatorial', observer)
    print 'equatorial2:', equatorial2


def get_test_runlist():
    """Get a runlist for the tests"""
    from atpy import Table
    table = Table()
    table.add_column('Run', [18373, 20581])
    table.add_column('Start_Time', ['2004-01-19 19:51:26', '2004-04-27 23:31:59'])
    table.add_column('Duration', [1580.0, 1682.0])
    table.add_column('GLON', [184.557, 359.346])
    table.add_column('GLAT', [-5.784, 0.410])
    return table


def test_compute_position_angles():
    observer = HESS()
    time_method = 'center'
    runs = get_test_runlist()
    position_angles = compute_position_angles(runs, observer, time_method)
    print position_angles


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

    # @note: alt increases where x increases, az increases where y increases
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


if __name__ == '__main__':
    # Some basic tests
    #test_HESS()
    test_transform()
    #test_compute_position_angles()


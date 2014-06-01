# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simulate source catalogs.
"""
from __future__ import print_function, division
import numpy as np
from numpy import sqrt, degrees, pi, arctan, arctan2, arcsin, exp
from numpy.random import uniform, normal
from astropy.table import Table, Column
from ...utils import coordinates as astrometry
from ...utils.const import d_sun_to_galactic_center
from ...utils.distributions import draw
from ...morphology.shapes import morph_types
from ..source import SNR, PWN, ModelPulsar
from .spatial import exponential, FaucherSpiral, r_range, z_range
from .velocity import v_range


__all__ = ['make_cat_cube',
           'make_cat_gal',
           'add_par_snr',
           'add_par_psr',
           'add_par_pwn',
           'add_par_obs',
           'add_cylindrical_coordinates',
           'add_observed_paramters'
           ]

def make_cat_cube(nsources=100, dimension=3, dmax=10,
                  luminosity_default=1,
                  extension_default=1):
    """Make a catalog of sources randomly distributed
    on a line, square or cube.
    """
    # Generate positions 1D, 2D, or 3D
    if dimension == 3:
        x = uniform(-dmax, dmax, nsources)
        y = uniform(-dmax, dmax, nsources)
        z = uniform(-dmax, dmax, nsources)
    elif dimension == 2:
        x = uniform(-dmax, dmax, nsources)
        y = uniform(-dmax, dmax, nsources)
        z = np.zeros(nsources)
    else:
        x = uniform(-dmax, dmax, nsources)
        y = np.zeros(nsources)
        z = np.zeros(nsources)

    luminosity = luminosity_default * np.ones(nsources)
    extension = extension_default * np.ones(nsources)

    table = Table()
    table['x'] = Column(x, unit='pc', description='Galactic cartesian coordinate')
    table['y'] = Column(y, unit='pc', description='Galactic cartesian coordinate')
    table['z'] = Column(z, unit='pc', description='Galactic cartesian coordinate')
    table['luminosity'] = Column(luminosity, description='Source luminosity')
    table['extension'] = Column(extension, unit='pc', description='Source physical radius')

    return table

'''
def make_cat_gauss_random(nsources=100, glon_sigma=30, glat_sigma=1,
                          extension_mean=0, extension_sigma=0.3,
                          flux_index=1, flux_min=10, flux_max=1000,
                          **kwargs):
    """Generate a catalog of Gaussian sources with random parameters.

    Default GLON, GLAT, EXTENSION, FLUX distributions
    are similar to what was observed by HESS.

    Useful for simulations of detection and fitting methods."""
    morph_type = np.array(['gauss2d']*nsources)
    glon = normal(0, glon_sigma, nsources) % 360
    glon_sym = np.where(glon < 180, glon, glon - 360)
    glat = normal(0, glat_sigma, nsources)
    sigma = normal(extension_mean, extension_sigma, nsources)
    sigma[sigma < 0] = 0
    ampl = draw(flux_min, flux_max, nsources, power_law,
                index=flux_index)

    names = ['morph_type', 'glon', 'glon_sym', 'glat', 'ampl', 'sigma']
    units = ['', 'deg', 'deg', 'deg', 'cm^-2 s^-1', 'deg']
    table = make_fits_table(locals(), names, units)
    return add_missing_morphology_columns(table)


def make_cat_gauss_grid(nside=3, sigma_min=0.05, flux_min=1e-11):
    """A test catalog for fitting which contains
    just a few Gaussians in a grid"""
    nsources = nside ** 2
    GLON = np.zeros(nsources)
    GLAT = np.zeros(nsources)
    sigma = np.zeros(nsources)
    flux = np.zeros(nsources)
    for a in range(nside):
        for b in range(nside):
            i = a + nside * b
            GLON[i] = a
            GLAT[i] = b
            sigma[i] = sigma_min * a
            flux[i] = flux_min * (2 ** b)

    # These columns are required so that to_image works:
    morph_type = np.array(['gauss2d'] * nsources)
    ampl = flux

    names = ['GLON', 'GLAT', 'morph_type'
             'sigma', 'flux', 'ampl']
    table = make_fits_table(locals(), names)
    return add_missing_morphology_columns(table)
'''

def make_cat_gal(nsources, rad_dis, vel_dis,
                 max_age, spiralarms=True, n_ISM=1,
                 **kwargs):
    """Make catalog of Galactic sources.

    Choose a radial distribution, a velocity distribution, the number
    of pulsars n_pulsars, the maximal age max_age[years] and the fraction
    of the individual morphtypes. There's an option spiralarms. If set on
    True a spiralarm modelling after Faucher&Kaspi is included.

    max_age and nsources effectively correspond to s SN rate:
    SN_rate = nsources / max_age
    """
    # Draw r and z values from the given distribution
    r = draw(0, r_range, nsources, rad_dis)
    z = draw(-z_range, z_range, nsources, exponential)

    # Draw values from velocity distribution
    v = draw(0, v_range, nsources, vel_dis)

    # Apply spiralarm modelling or not
    if spiralarms:
        theta, spiralarm, dx, dy = FaucherSpiral()(r)
    else:
        theta = uniform(0, 2 * pi, nsources)
        dx = dy = np.zeros(nsources)
        spiralarm = FaucherSpiral().spiralarms[3 * np.ones((nsources,), dtype=np.int)]

    # Compute cartesian coordinates
    x, y = astrometry.cartesian(r, theta, dx=dx, dy=dy)

    # Draw random values for the age
    age = uniform(0, max_age, nsources)

    # Draw random direction of initial velocity
    theta = uniform(0, pi, x.size)
    phi = uniform(0, 2 * pi, x.size)

    # Set environment interstellar density
    n_ISM = n_ISM * np.ones(nsources)

    # Compute galactic coordinate and distance to observer
    glon, glat, distance = astrometry.galactic(x, y, z)

    # Compute Equatorial position
    RA, DEC = astrometry.sky_to_sky(glon, glat, 'galactic', 'icrs')

    # Compute new position
    x, y, z, vx, vy, vz = astrometry.motion_since_birth(x, y, z, v, age, theta, phi)

    # Compute projected velocity
    v_glon, v_glat = astrometry.spherical_velocity(x, y, z, vx, vy, vz)

    # For now we only simulate shell-type SNRs.
    # Later we might want to simulate certain fractions of object classes
    # index = random_integers(0, 0, nsources)
    index = 2 * np.ones(nsources, dtype=np.int)
    morph_type = np.array(list(morph_types.keys()))[index]

    table = Table()
    table['x_birth'] = Column(x, unit='kpc')
    table['y_birth'] = Column(y, unit='kpc')
    table['z_birth'] = Column(z, unit='kpc')
    table['glon_birth'] = Column(glon, unit='deg')
    table['glat_birth'] = Column(glat, unit='deg')
    table['x'] = Column(x, unit='kpc')
    table['y'] = Column(y, unit='kpc')
    table['z'] = Column(z, unit='kpc')
    table['GLON'] = Column(glon, unit='deg')
    table['GLAT'] = Column(glat, unit='deg')
    table['RA'] = Column(RA, unit='deg')
    table['DEC'] = Column(DEC, unit='deg')
    table['distance'] = Column(distance, unit='kpc')
    table['age'] = Column(age, unit='years')
    table['n_ISM'] = Column(n_ISM, unit='cm^-3')
    table['spiralarm'] = spiralarm
    table['morph_type'] = morph_type
    table['v_glon'] = Column(v_glon, unit='1e-6 deg yr^-1')
    table['v_glat'] = Column(v_glat, unit='1e-6 deg yr^-1')
    table['v_abs'] = Column(v, unit='km s^-1')

    return table


def add_par_snr(table, E_SN=1e51):
    """Adds SNR parameters to the table.
    """
    # Read relevant columns
    age = table['age']
    n_ISM = table['n_ISM']

    # Compute properties
    snr = SNR(E_SN=E_SN, n_ISM=n_ISM)
    E_SN = snr.E_SN * np.ones_like(age)
    r_out = snr.r_out(age)
    r_in = snr.r_in(age)
    L_SNR = snr.L(age)

    # Add columns to table    
    table['E_SN'] = Column(E_SN, unit='erg', description='SNR kinetic energy')
    table['r_out'] = Column(r_out, unit='pc', description='SNR outer radius')
    table['r_in'] = Column(r_in, unit='pc', description='SNR inner radius')
    table['L_SNR'] = Column(L_SNR, unit='s^-1', description='SNR luminosity')

    return table


def add_par_psr(table, B_mean=12.05, B_stdv=0.55,
                P_mean=0.3, P_stdv=0.15):
    """Adds pulsar parameters to the table.

    For the initial normal distribution of period and logB can exist the following
    Parameters: B_mean=12.05[log Gauss], B_stdv=0.55, P_mean=0.3[s], P_stdv=0.15
    """
    # Read relevant columns
    age = table['age']

    # Draw the initial values for the period and magnetic field
    P_dist = lambda x: exp(-0.5 * ((x - P_mean) / P_stdv) ** 2)
    P0_birth = draw(0, 2, len(table), P_dist)
    logB = normal(B_mean, B_stdv, len(table))

    # Compute birth properties
    psr = ModelPulsar(P0_birth, logB)
    P1_birth = psr.Pdot(0)
    Tau0 = psr.tau_0
    L0_PSR = psr.L(0)

    # Compute current properties
    P0 = psr.P(age)
    P1 = psr.Pdot(age)
    CharAge = psr.CharAge(age)
    L_PSR = psr.L(age)

    # Add columns to table
    # TODO: Name all columns as in ATNF catalog
    table['P0'] = Column(P0, unit='s', description='Pulsar period')
    table['P1'] = Column(P1, unit='', description='Pulsar period derivative')
    table['P0_birth'] = Column(P0_birth, unit='s', description='Pulsar birth period')
    table['P1_birth'] = Column(P1_birth, unit='', description='Pulsar birth period derivative')
    table['CharAge'] = Column(CharAge, unit='yr', description='Pulsar characteristic age')
    table['Tau0'] = Column(Tau0, unit='yr')
    table['L_PSR'] = Column(L_PSR, unit='erg s^-1')
    table['L0_PSR'] = Column(L0_PSR, unit='erg s^-1')
    table['logB'] = Column(logB, unit='Gauss')

    return table


def add_par_pwn(table):
    """Adds PWN parameters to the table.
    """
    # Read relevant columns
    age = table['age']
    E_SN = table['E_SN']
    n_ISM = table['n_ISM']
    P0_birth = table['P0_birth']
    logB = table['logB']

    # Compute properties
    pwn = PWN(ModelPulsar(P0_birth, logB),
              SNR(E_SN=E_SN, n_ISM=n_ISM))
    r_out_pwn = pwn.r(age)
    L_PWN = pwn.L(age)

    # Add columns to table
    table['r_out_PWN'] = Column(r_out_pwn, unit='pc', description='PWN outer radius')
    table['L_PWN'] = Column(L_PWN, unit='s^-1', description='PWN luminosity above 1 TeV')

    return table


def add_par_obs(table):
    """Adds observed source parameters to the table.
    """
    # Read relevant columns
    distance = table['distance']
    r_in = table['r_in']
    r_out = table['r_out']
    r_out_PWN = table['r_out_PWN']
    L_SNR = table['L_SNR']
    L_PSR = table['L_PSR']
    L_PWN = table['L_PWN']

    # Compute properties
    ext_in_SNR = astrometry.radius_to_angle(r_in, distance)
    ext_out_SNR = astrometry.radius_to_angle(r_out, distance)
    ext_out_PWN = astrometry.radius_to_angle(r_out_PWN, distance)

    # Ellipse parameters not used for now
    theta = pi / 2 * np.ones(len(table))  # Position angle?
    epsilon = np.zeros(len(table))  # Ellipticity?

    S_SNR = astrometry.luminosity_to_flux(L_SNR, distance)
    # Ld2_PSR = astrometry.luminosity_to_flux(L_PSR, distance)
    Ld2_PSR = L_PSR / distance ** 2
    S_PWN = astrometry.luminosity_to_flux(L_PWN, distance)

    # Add columns
    table['ext_in_SNR'] = Column(ext_in_SNR, unit='deg')
    table['ext_out_SNR'] = Column(ext_out_SNR, unit='deg')
    table['ext_out_PWN'] = Column(ext_out_PWN, unit='deg')
    table['theta'] = Column(theta, unit='rad')
    table['epsilon'] = Column(epsilon, unit='')
    table['S_SNR'] = Column(S_SNR, unit='cm^-2 s^-1')
    table['Ld2_PSR'] = Column(Ld2_PSR, unit='erg s^-1 kpc^-2')
    table['S_PWN'] = Column(S_PWN, unit='cm^-2 s^-1')

    return table


def add_cylindrical_coordinates(table):
    """Adds two colums with r and phi to a table
    containing cartesion coordinates x and y.
    """
    x = table['x']
    y = table['y']

    r = sqrt(x ** 2 + y ** 2)
    phi = degrees(arctan2(y, x))

    table['r'] =Column(r, unit='pc', description='Galactic cylindrical coordinate')
    table['phi'] = Column(phi, unit='deg', description='Galactic cylindrical coordinate')

    return table


def add_observed_parameters(table, obs_pos=[d_sun_to_galactic_center, 0, 0]):
    """For a given observer position (default: earth)
    add observed parameters to the
    table for given physical parameters.

    Input parameters:
    x, y, z, extension, luminosity

    Output parameters:
    distance, glon, glat, flux, angular_extension

    Position of observer in cartesian coordinates.
    Center of galaxy as origin, x-axis goes trough sun.
    """
    # Get data
    x = table['x']
    y = table['y']
    z = table['z']
    extension = table['extension']
    luminosity = table['luminosity']

    # Subtract P.o.O. (Position of Observer)
    x = x - obs_pos[0]
    y = y - obs_pos[1]
    z = z - obs_pos[2]

    # Compute observable parameters
    # Note that the formula for angular extension is only an approximation
    # TODO: add correct angular extension as extra column
    distance = sqrt(x ** 2 + y ** 2 + z ** 2)
    glon = degrees(arctan2(y, x))
    glat = degrees(arcsin(z / distance))
    flux = luminosity / (4 * pi * distance ** 2)
    angular_extension = degrees(arctan(extension / distance))

    # Add columns to table
    table['distance'] = Column(distance, unit='pc', description='Distance observer to source center')
    table['glon'] = Column(glon, unit='deg', description='Galactic longitude')
    table['glat'] = Column(glat, unit='deg', description='Galactic latitude')
    table['flux'] = Column(flux, unit='', description='Source flux')
    table['angular_extension'] = Column(angular_extension, unit='deg',
                                        description='Source angular radius (i.e. half-diameter)')

    return table

# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simulate source catalogs.
"""
from __future__ import print_function, division
import numpy as np
from numpy import degrees, pi, arctan, exp
from numpy.random import uniform, normal
from astropy.table import Table, Column
from astropy.units import Quantity
from astropy.coordinates import SkyCoord

from ...utils import coordinates as astrometry
from ...utils.const import d_sun_to_galactic_center
from ...utils.distributions import draw, pdf
from ...morphology.shapes import morph_types
from ...catalog.utils import as_quantity
from ..source import SNR, SNRTrueloveMcKee, PWN, Pulsar
from ..population import Exponential, FaucherSpiral, RMIN, RMAX, ZMIN, ZMAX, radial_distributions
from ..population import VMIN, VMAX, velocity_distributions


__all__ = ['make_cat_cube',
           'make_base_catalog_galactic',
           'add_snr_parameters',
           'add_pulsar_parameters',
           'add_pwn_parameters',
           'add_observed_source_parameters',
           'add_observed_parameters',
           ]


def make_cat_cube(n_sources=100, dimension=3, dmax=10,
                  luminosity_default=1,
                  extension_default=1):
    """Make a catalog of sources randomly distributed
    on a line, square or cube.
    """
    # Generate positions 1D, 2D, or 3D
    if dimension == 3:
        x = uniform(-dmax, dmax, n_sources)
        y = uniform(-dmax, dmax, n_sources)
        z = uniform(-dmax, dmax, n_sources)
    elif dimension == 2:
        x = uniform(-dmax, dmax, n_sources)
        y = uniform(-dmax, dmax, n_sources)
        z = np.zeros(n_sources)
    else:
        x = uniform(-dmax, dmax, n_sources)
        y = np.zeros(n_sources)
        z = np.zeros(n_sources)

    luminosity = luminosity_default * np.ones(n_sources)
    extension = extension_default * np.ones(n_sources)

    table = Table()
    table['x'] = Column(x, unit='pc', description='Galactic cartesian coordinate')
    table['y'] = Column(y, unit='pc', description='Galactic cartesian coordinate')
    table['z'] = Column(z, unit='pc', description='Galactic cartesian coordinate')
    table['luminosity'] = Column(luminosity, description='Source luminosity')
    table['extension'] = Column(extension, unit='pc', description='Source physical radius')

    return table

'''
def make_cat_gauss_random(n_sources=100, glon_sigma=30, glat_sigma=1,
                          extension_mean=0, extension_sigma=0.3,
                          flux_index=1, flux_min=10, flux_max=1000):
    """Generate a catalog of Gaussian sources with random parameters.

    Default GLON, GLAT, EXTENSION, FLUX distributions
    are similar to what was observed by HESS.

    Useful for simulations of detection and fitting methods."""
    morph_type = np.array(['gauss2d']*n_sources)
    glon = normal(0, glon_sigma, n_sources) % 360
    glon_sym = np.where(glon < 180, glon, glon - 360)
    glat = normal(0, glat_sigma, n_sources)
    sigma = normal(extension_mean, extension_sigma, n_sources)
    sigma[sigma < 0] = 0
    ampl = draw(flux_min, flux_max, n_sources, power_law,
                index=flux_index)

    names = ['morph_type', 'glon', 'glon_sym', 'glat', 'ampl', 'sigma']
    units = ['', 'deg', 'deg', 'deg', 'cm^-2 s^-1', 'deg']
    table = make_fits_table(locals(), names, units)
    return add_missing_morphology_columns(table)


def make_cat_gauss_grid(nside=3, sigma_min=0.05, flux_min=1e-11):
    """A test catalog for fitting which contains
    just a few Gaussians in a grid"""
    n_sources = nside ** 2
    GLON = np.zeros(n_sources)
    GLAT = np.zeros(n_sources)
    sigma = np.zeros(n_sources)
    flux = np.zeros(n_sources)
    for a in range(nside):
        for b in range(nside):
            i = a + nside * b
            GLON[i] = a
            GLAT[i] = b
            sigma[i] = sigma_min * a
            flux[i] = flux_min * (2 ** b)

    # These columns are required so that to_image works:
    morph_type = np.array(['gauss2d'] * n_sources)
    ampl = flux

    names = ['GLON', 'GLAT', 'morph_type'
             'sigma', 'flux', 'ampl']
    table = make_fits_table(locals(), names)
    return add_missing_morphology_columns(table)
'''


def make_base_catalog_galactic(n_sources, rad_dis='YK04', vel_dis='H05',
                               max_age=Quantity(1E6, 'yr'),
                               spiralarms=True, n_ISM=Quantity(1, 'cm^-3')):
    """
    Make a catalog of Galactic sources, with basic parameters like position, age and
    proper velocity.

    Choose a radial distribution, a velocity distribution, the number
    of pulsars n_pulsars, the maximal age max_age[years] and the fraction
    of the individual morphtypes. There's an option spiralarms. If set on
    True a spiralarm modelling after Faucher&Kaspi is included.

    max_age and n_sources effectively correspond to s SN rate:
    SN_rate = n_sources / max_age

    Parameters
    ----------
    n_sources : int
        Number of sources to simulate.
    rad_dis : callable
        Radial surface density distribution of sources.
    vel_dis : callable
        Proper motion velocity distribution of sources.
    max_age : `~astropy.units.Quantity`
        Maximal age of the source
    spiralarms : bool
        Include a spiralarm model in the catalog.
    n_ISM : `~astropy.units.Quantity`
        Density of the interstellar medium.

    Returns
    -------
    table : `~astropy.table.Table`
        Catalog of simulated source positions and proper velocities.
    """
    if isinstance(rad_dis, str):
        rad_dis = radial_distributions[rad_dis]

    if isinstance(vel_dis, str):
        vel_dis = velocity_distributions[vel_dis]

    # Draw r and z values from the given distribution
    r = Quantity(draw(RMIN.value, RMAX.value, n_sources, pdf(rad_dis())), 'kpc')
    z = Quantity(draw(ZMIN.value, ZMAX.value, n_sources, Exponential()), 'kpc')

    # Draw values from velocity distribution
    v = Quantity(draw(VMIN.value, VMAX.value, n_sources, vel_dis()), 'km/s')

    # Apply spiralarm modelling or not
    if spiralarms:
        r, theta, spiralarm = FaucherSpiral()(r)
    else:
        theta = Quantity(uniform(0, 2 * pi, n_sources), 'rad')
        spiralarm = None

    # Compute cartesian coordinates
    x, y = astrometry.cartesian(r, theta)

    # Draw random values for the age
    age = Quantity(uniform(0, max_age, n_sources), 'yr')

    # Draw random direction of initial velocity
    theta = Quantity(uniform(0, pi, x.size), 'rad')
    phi = Quantity(uniform(0, 2 * pi, x.size), 'rad')

    # Set environment interstellar density
    n_ISM = n_ISM * np.ones(n_sources)

    # Compute new position
    # TODO: uncomment this for the moment ... it changes `x` from parsec
    # to km which it shouldn't.
    dx, dy, dz, vx, vy, vz = astrometry.motion_since_birth(v, age, theta, phi)

    # Add displacemt to birth position
    x += dx.to('kpc')
    y += dy.to('kpc')
    z += dz.to('kpc')

    # For now we only simulate shell-type SNRs.
    # Later we might want to simulate certain fractions of object classes
    # index = random_integers(0, 0, n_sources)
    index = 2 * np.ones(n_sources, dtype=np.int)
    morph_type = np.array(list(morph_types.keys()))[index]

    table = Table()
    table['x_birth'] = Column(x, unit='kpc')
    table['y_birth'] = Column(y, unit='kpc')
    table['z_birth'] = Column(z, unit='kpc')
    table['x'] = Column(x, unit='kpc')
    table['y'] = Column(y, unit='kpc')
    table['z'] = Column(z, unit='kpc')
    table['vx'] = Column(vx.to('km/s'), unit='km/s')
    table['vy'] = Column(vy.to('km/s'), unit='km/s')
    table['vz'] = Column(vz.to('km/s'), unit='km/s')

    table['age'] = Column(age, unit='yr')
    table['n_ISM'] = Column(n_ISM, unit='cm^-3')
    table['spiralarm'] = spiralarm
    table['morph_type'] = morph_type
    table['v_abs'] = Column(v, unit='km/s')
    return table


def add_snr_parameters(table):
    """Adds SNR parameters to the table.
    """
    # Read relevant columns
    age = as_quantity(table['age'])
    n_ISM = as_quantity(table['n_ISM'])

    # Compute properties
    snr = SNR(n_ISM=n_ISM)
    E_SN = snr.e_sn * np.ones(len(table))
    r_out = snr.radius(age)
    r_in = snr.radius_inner(age)
    L_SNR = snr.luminosity_tev(age)

    # Add columns to table
    table['E_SN'] = Column(E_SN, unit='erg', description='SNR kinetic energy')
    table['r_out'] = Column(r_out, unit='pc', description='SNR outer radius')
    table['r_in'] = Column(r_in, unit='pc', description='SNR inner radius')
    table['L_SNR'] = Column(L_SNR, unit='ph s^-1', description='SNR luminosity')
    return table


def add_pulsar_parameters(table, B_mean=12.05, B_stdv=0.55,
                P_mean=0.3, P_stdv=0.15):
    """Adds pulsar parameters to the table.

    For the initial normal distribution of period and logB can exist the following
    Parameters: B_mean=12.05[log Gauss], B_stdv=0.55, P_mean=0.3[s], P_stdv=0.15
    """
    # Read relevant columns
    age = as_quantity(table['age'])

    # Draw the initial values for the period and magnetic field
    P_dist = lambda x: exp(-0.5 * ((x - P_mean) / P_stdv) ** 2)
    P0_birth = Quantity(draw(0, 2, len(table), P_dist), 's')
    logB = normal(B_mean, B_stdv, len(table))

    # Set up pulsar model
    psr = Pulsar(P0_birth, logB)

    # Add columns to table
    # TODO: Name all columns as in ATNF catalog
    table['P0'] = Column(psr.period(age), unit='s', description='Pulsar period')
    table['P1'] = Column(psr.period_dot(age), unit='', description='Pulsar period derivative')
    table['P0_birth'] = Column(P0_birth, unit='s', description='Pulsar birth period')
    table['P1_birth'] = Column(psr.P_dot_0, unit='', description='Pulsar birth period derivative')
    table['CharAge'] = Column(psr.tau(age), unit='yr', description='Pulsar characteristic age')
    table['Tau0'] = Column(psr.tau_0, unit='yr')
    table['L_PSR'] = Column(psr.luminosity_spindown(age), unit='erg s^-1')
    table['L0_PSR'] = Column(psr.L_0, unit='erg s^-1')
    table['logB'] = Column(logB, unit='Gauss')
    return table


def add_pwn_parameters(table):
    """Adds PWN parameters to the table.
    """
    # Read relevant columns
    age = as_quantity(table['age'])
    E_SN = as_quantity(table['E_SN'])
    n_ISM = as_quantity(table['n_ISM'])
    P0_birth = as_quantity(table['P0_birth'])
    logB = table['logB']

    # Compute properties
    pwn = PWN(Pulsar(P0_birth, logB),
              SNRTrueloveMcKee(e_sn=E_SN, n_ISM=n_ISM))
    r_out_pwn = pwn.radius(age)
    L_PWN = pwn.luminosity_tev(age)

    # Add columns to table
    table['r_out_PWN'] = Column(r_out_pwn, unit='pc', description='PWN outer radius')
    table['L_PWN'] = Column(L_PWN, unit='erg', description='PWN luminosity above 1 TeV')
    return table


def add_observed_source_parameters(table):
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
    x, y, z = as_quantity(table['x'], table['y'], table['z'])
    vx, vy, vz = as_quantity(table['vx'], table['vy'], table['vz'])

    distance, glon, glat = astrometry.galactic(x, y, z)

    # Compute projected velocity
    v_glon, v_glat = astrometry.velocity_glon_glat(x, y, z, vx, vy, vz)

    coordinate = SkyCoord(glon, glat, unit='deg', frame='galactic').transform_to('icrs')
    ra, dec = coordinate.ra.deg, coordinate.dec.deg

    # Add columns to table
    table['distance'] = Column(distance, unit='pc',
                               description='Distance observer to source center')
    table['GLON'] = Column(glon, unit='deg', description='Galactic longitude')
    table['GLAT'] = Column(glat, unit='deg', description='Galactic latitude')
    table['VGLON'] = Column(v_glon.to('deg/Myr'), unit='deg/Myr',
                            description='Velocity in Galactic longitude')
    table['VGLAT'] = Column(v_glat.to('deg/Myr'), unit='deg/Myr',
                            description='Velocity in Galactic latitude')
    table['RA'] = Column(ra, unit='deg')
    table['DEC'] = Column(dec, unit='deg')

    try:
        luminosity = table['luminosity']
        flux = astrometry.luminosity_to_flux(luminosity, distance)
        table['flux'] = Column(flux.value, unit=flux.unit, description='Source flux')
    except KeyError:
        pass

    try:
        extension = table['extension']
        angular_extension = degrees(arctan(extension / distance))
        table['angular_extension'] = Column(angular_extension, unit='deg',
                                            description='Source angular radius (i.e. half-diameter)')
    except KeyError:
        pass

    return table

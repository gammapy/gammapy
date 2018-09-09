# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simulate source catalogs."""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from ...extern import six
from astropy.table import Table, Column
from astropy.units import Quantity
from astropy.coordinates import SkyCoord, spherical_to_cartesian
from ...utils import coordinates as astrometry
from ...utils.coordinates import D_SUN_TO_GALACTIC_CENTER
from ...utils.distributions import draw, pdf
from ...utils.random import sample_sphere, sample_sphere_distance, get_random_state
from ..source import SNR, SNRTrueloveMcKee, PWN, Pulsar
from ..population.spatial import (
    Exponential,
    FaucherSpiral,
    RMIN,
    RMAX,
    ZMIN,
    ZMAX,
    radial_distributions,
)
from ..population.velocity import VMIN, VMAX, velocity_distributions

__all__ = [
    "make_catalog_random_positions_cube",
    "make_catalog_random_positions_sphere",
    "make_base_catalog_galactic",
    "add_snr_parameters",
    "add_pulsar_parameters",
    "add_pwn_parameters",
    "add_observed_source_parameters",
    "add_observed_parameters",
]


def make_catalog_random_positions_cube(
    size=100, dimension=3, dmax=10, random_state="random-seed"
):
    """Make a catalog of sources randomly distributed on a line, square or cube.

    TODO: is this useful enough for general use or should we hide it as an
      internal method to generate test datasets?

    Parameters
    ----------
    size : int, optional
        Number of sources
    dimension : int, optional
        Number of dimensions
    dmax : int, optional
        Maximum distance in pc.
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.

    Returns
    -------
    catalog : `~astropy.table.Table`
        Source catalog with columns:
    """
    random_state = get_random_state(random_state)

    # Generate positions 1D, 2D, or 3D
    if dimension == 3:
        x = random_state.uniform(-dmax, dmax, size)
        y = random_state.uniform(-dmax, dmax, size)
        z = random_state.uniform(-dmax, dmax, size)
    elif dimension == 2:
        x = random_state.uniform(-dmax, dmax, size)
        y = random_state.uniform(-dmax, dmax, size)
        z = np.zeros_like(x)
    else:
        x = random_state.uniform(-dmax, dmax, size)
        y = np.zeros_like(x)
        z = np.zeros_like(x)

    table = Table()
    table["x"] = Column(x, unit="pc", description="Galactic cartesian coordinate")
    table["y"] = Column(y, unit="pc", description="Galactic cartesian coordinate")
    table["z"] = Column(z, unit="pc", description="Galactic cartesian coordinate")

    return table


def make_catalog_random_positions_sphere(
    size, center="Earth", distance=Quantity([0, 1], "Mpc"), random_state="random-seed"
):
    """Sample random source locations in a sphere.

    This can be used to generate an isotropic source population
    to represent extra-galactic sources.

    Parameters
    ----------
    size : int
        Number of sources
    center : {'Earth', 'Milky Way'}
        Sphere center
    distance : `~astropy.units.Quantity` tuple
        Distance min / max range.
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.

    Returns
    -------
    catalog : `~astropy.table.Table`
        Source catalog with columns:

        - RAJ2000, DEJ2000 (deg)
        - GLON, GLAT (deg)
        - Distance (Mpc)
    """
    random_state = get_random_state(random_state)

    lon, lat = sample_sphere(size, random_state=random_state)
    radius = sample_sphere_distance(
        distance[0], distance[1], size, random_state=random_state
    )

    # TODO: it shouldn't be necessary here to convert to cartesian ourselves ...
    x, y, z = spherical_to_cartesian(radius, lat, lon)
    pos = SkyCoord(x, y, z, frame="galactocentric", representation="cartesian")

    if center == "Milky Way":
        pass
    elif center == "Earth":
        # TODO: add shift Galactic center -> Earth
        raise NotImplementedError
    else:
        msg = "Invalid center: {}\n".format(center)
        msg += "Choose one of: Earth, Milky Way"
        raise ValueError(msg)

    table = Table()
    table.meta["center"] = center

    icrs = pos.transform_to("icrs")
    table["RAJ2000"] = icrs.ra.to("deg")
    table["DEJ2000"] = icrs.dec.to("deg")

    galactic = icrs.transform_to("galactic")
    table["GLON"] = galactic.l.to("deg")
    table["GLAT"] = galactic.b.to("deg")

    table["Distance"] = icrs.distance.to("Mpc")

    return table


def make_base_catalog_galactic(
    n_sources,
    rad_dis="YK04",
    vel_dis="H05",
    max_age=Quantity(1e6, "yr"),
    spiralarms=True,
    n_ISM=Quantity(1, "cm-3"),
    random_state="random-seed",
):
    """Make a catalog of Galactic sources, with basic source parameters.

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
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.

    Returns
    -------
    table : `~astropy.table.Table`
        Catalog of simulated source positions and proper velocities.
    """
    random_state = get_random_state(random_state)

    if isinstance(rad_dis, six.string_types):
        rad_dis = radial_distributions[rad_dis]

    if isinstance(vel_dis, six.string_types):
        vel_dis = velocity_distributions[vel_dis]

    # Draw random values for the age
    age = random_state.uniform(0, max_age.to("yr").value, n_sources)
    age = Quantity(age, "yr")

    # Draw r and z values from the given distribution
    r = draw(
        RMIN.to("kpc").value,
        RMAX.to("kpc").value,
        n_sources,
        pdf(rad_dis()),
        random_state=random_state,
    )
    r = Quantity(r, "kpc")

    z = draw(
        ZMIN.to("kpc").value,
        ZMAX.to("kpc").value,
        n_sources,
        Exponential(),
        random_state=random_state,
    )
    z = Quantity(z, "kpc")

    # Apply spiralarm modelling or not
    if spiralarms:
        r, theta, spiralarm = FaucherSpiral()(r, random_state=random_state)
    else:
        theta = Quantity(random_state.uniform(0, 2 * np.pi, n_sources), "rad")
        spiralarm = None

    # Compute cartesian coordinates
    x, y = astrometry.cartesian(r, theta)

    # Draw values from velocity distribution
    v = draw(
        VMIN.to("km/s").value,
        VMAX.to("km/s").value,
        n_sources,
        vel_dis(),
        random_state=random_state,
    )
    v = Quantity(v, "km/s")

    # Draw random direction of initial velocity
    theta = Quantity(random_state.uniform(0, np.pi, x.size), "rad")
    phi = Quantity(random_state.uniform(0, 2 * np.pi, x.size), "rad")

    # Compute new position
    dx, dy, dz, vx, vy, vz = astrometry.motion_since_birth(v, age, theta, phi)

    # Add displacement to birth position
    x_moved = x + dx
    y_moved = y + dy
    z_moved = z + dz

    # Set environment interstellar density
    n_ISM = n_ISM * np.ones(n_sources)

    table = Table()
    table["age"] = Column(age, unit="yr", description="Age of the source")
    table["n_ISM"] = Column(
        n_ISM, unit="cm-3", description="Interstellar medium density"
    )
    if spiralarms:
        table["spiralarm"] = Column(spiralarm, description="Which spiralarm?")

    table["x_birth"] = Column(
        x, unit="kpc", description="Galactocentric x coordinate at birth"
    )
    table["y_birth"] = Column(
        y, unit="kpc", description="Galactocentric y coordinate at birth"
    )
    table["z_birth"] = Column(
        z, unit="kpc", description="Galactocentric z coordinate at birth"
    )

    table["x"] = Column(
        x_moved.to("kpc"), unit="kpc", description="Galactocentric x coordinate"
    )
    table["y"] = Column(
        y_moved.to("kpc"), unit="kpc", description="Galactocentric y coordinate"
    )
    table["z"] = Column(
        z_moved.to("kpc"), unit="kpc", description="Galactocentric z coordinate"
    )

    table["vx"] = Column(
        vx.to("km/s"), unit="km/s", description="Galactocentric velocity in x direction"
    )
    table["vy"] = Column(
        vy.to("km/s"), unit="km/s", description="Galactocentric velocity in y direction"
    )
    table["vz"] = Column(
        vz.to("km/s"), unit="km/s", description="Galactocentric velocity in z direction"
    )
    table["v_abs"] = Column(
        v, unit="km/s", description="Galactocentric velocity (absolute)"
    )

    return table


def add_snr_parameters(table):
    """Add SNR parameters to the table."""
    # Read relevant columns
    age = table["age"].quantity
    n_ISM = table["n_ISM"].quantity

    # Compute properties
    snr = SNR(n_ISM=n_ISM)
    E_SN = snr.e_sn * np.ones(len(table))
    r_out = snr.radius(age)
    r_in = snr.radius_inner(age)
    L_SNR = snr.luminosity_tev(age)

    # Add columns to table
    table["E_SN"] = Column(E_SN, unit="erg", description="SNR kinetic energy")
    table["r_out"] = Column(r_out, unit="pc", description="SNR outer radius")
    table["r_in"] = Column(r_in, unit="pc", description="SNR inner radius")
    table["L_SNR"] = Column(L_SNR, unit="s-1", description="SNR luminosity")
    return table


def add_pulsar_parameters(
    table,
    B_mean=12.05,
    B_stdv=0.55,
    P_mean=0.3,
    P_stdv=0.15,
    random_state="random-seed",
):
    """Add pulsar parameters to the table.

    For the initial normal distribution of period and logB can exist the following
    Parameters: B_mean=12.05[log Gauss], B_stdv=0.55, P_mean=0.3[s], P_stdv=0.15

    Parameters
    ----------
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.

    """
    random_state = get_random_state(random_state)
    # Read relevant columns
    age = table["age"].quantity

    # Draw the initial values for the period and magnetic field
    def p_dist(x):
        return np.exp(-0.5 * ((x - P_mean) / P_stdv) ** 2)

    p0_birth = draw(0, 2, len(table), p_dist, random_state=random_state)
    p0_birth = Quantity(p0_birth, "s")

    logB = random_state.normal(B_mean, B_stdv, len(table))

    # Compute pulsar parameters
    psr = Pulsar(p0_birth, logB)
    p0 = psr.period(age)
    p1 = psr.period_dot(age)
    p1_birth = psr.P_dot_0
    tau = psr.tau(age)
    tau_0 = psr.tau_0
    l_psr = psr.luminosity_spindown(age)
    l0_psr = psr.L_0

    # Add columns to table
    table["P0"] = Column(p0, unit="s", description="Pulsar period")
    table["P1"] = Column(p1, unit="", description="Pulsar period derivative")
    table["P0_birth"] = Column(p0_birth, unit="s", description="Pulsar birth period")
    table["P1_birth"] = Column(
        p1_birth, unit="", description="Pulsar birth period derivative"
    )
    table["CharAge"] = Column(tau, unit="yr", description="Pulsar characteristic age")
    table["Tau0"] = Column(tau_0, unit="yr")
    table["L_PSR"] = Column(l_psr, unit="erg s-1")
    table["L0_PSR"] = Column(l0_psr, unit="erg s-1")
    table["logB"] = Column(logB, unit="Gauss")
    return table


def add_pwn_parameters(table):
    """Add PWN parameters to the table."""
    # Some of the computations (specifically `pwn.radius`) aren't vectorised
    # across all parameters; so here we loop over source parameters explicitly

    results = []

    for idx in range(len(table)):
        age = table["age"].quantity[idx]
        E_SN = table["E_SN"].quantity[idx]
        n_ISM = table["n_ISM"].quantity[idx]
        P0_birth = table["P0_birth"].quantity[idx]
        logB = table["logB"][idx]

        # Compute properties
        pulsar = Pulsar(P0_birth, logB)
        snr = SNRTrueloveMcKee(e_sn=E_SN, n_ISM=n_ISM)
        pwn = PWN(pulsar, snr)
        r_out_pwn = pwn.radius(age).to("pc").value
        L_PWN = pwn.luminosity_tev(age).to("erg").value
        results.append(dict(r_out_pwn=r_out_pwn, L_PWN=L_PWN))

    # Add columns to table
    table["r_out_PWN"] = Column(
        [_["r_out_pwn"] for _ in results], unit="pc", description="PWN outer radius"
    )
    table["L_PWN"] = Column(
        [_["L_PWN"] for _ in results],
        unit="erg",
        description="PWN luminosity above 1 TeV",
    )
    return table


def add_observed_source_parameters(table):
    """Add observed source parameters to the table."""
    # Read relevant columns
    distance = table["distance"]
    r_in = table["r_in"]
    r_out = table["r_out"]
    r_out_PWN = table["r_out_PWN"]
    L_SNR = table["L_SNR"]
    L_PSR = table["L_PSR"]
    L_PWN = table["L_PWN"]

    # Compute properties
    ext_in_SNR = astrometry.radius_to_angle(r_in, distance)
    ext_out_SNR = astrometry.radius_to_angle(r_out, distance)
    ext_out_PWN = astrometry.radius_to_angle(r_out_PWN, distance)

    # Ellipse parameters not used for now
    theta = np.pi / 2 * np.ones(len(table))  # Position angle?
    epsilon = np.zeros(len(table))  # Ellipticity?

    S_SNR = astrometry.luminosity_to_flux(L_SNR, distance)
    # Ld2_PSR = astrometry.luminosity_to_flux(L_PSR, distance)
    Ld2_PSR = L_PSR / distance ** 2
    S_PWN = astrometry.luminosity_to_flux(L_PWN, distance)

    # Add columns
    table["ext_in_SNR"] = Column(ext_in_SNR, unit="deg")
    table["ext_out_SNR"] = Column(ext_out_SNR, unit="deg")
    table["ext_out_PWN"] = Column(ext_out_PWN, unit="deg")
    table["theta"] = Column(theta, unit="rad")
    table["epsilon"] = Column(epsilon, unit="")
    table["S_SNR"] = Column(S_SNR, unit="cm-2 s-1")
    table["Ld2_PSR"] = Column(Ld2_PSR, unit="erg s-1 kpc-2")
    table["S_PWN"] = Column(S_PWN, unit="cm-2 s-1")
    return table


def add_observed_parameters(table, obs_pos=None):
    """Add observable parameters (such as sky position or distance).

    Input table columns: x, y, z, extension, luminosity

    Output table columns: distance, glon, glat, flux, angular_extension

    Position of observer in cartesian coordinates.
    Center of galaxy as origin, x-axis goes trough sun.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Input table
    obs_pos : tuple or None
        Observation position (X, Y, Z) in Galactocentric coordinates (default: Earth)

    Returns
    -------
    table : `~astropy.table.Table`
        Modified input table with columns added
    """
    obs_pos = obs_pos or [D_SUN_TO_GALACTIC_CENTER, 0, 0]

    # Get data
    x, y, z = table["x"].quantity, table["y"].quantity, table["z"].quantity
    vx, vy, vz = table["vx"].quantity, table["vy"].quantity, table["vz"].quantity

    distance, glon, glat = astrometry.galactic(x, y, z, obs_pos=obs_pos)

    # Compute projected velocity
    v_glon, v_glat = astrometry.velocity_glon_glat(x, y, z, vx, vy, vz)

    coordinate = SkyCoord(glon, glat, unit="deg", frame="galactic").transform_to("icrs")
    ra, dec = coordinate.ra.deg, coordinate.dec.deg

    # Add columns to table
    table["distance"] = Column(
        distance, unit="pc", description="Distance observer to source center"
    )
    table["GLON"] = Column(glon, unit="deg", description="Galactic longitude")
    table["GLAT"] = Column(glat, unit="deg", description="Galactic latitude")
    table["VGLON"] = Column(
        v_glon.to("deg/Myr"),
        unit="deg/Myr",
        description="Velocity in Galactic longitude",
    )
    table["VGLAT"] = Column(
        v_glat.to("deg/Myr"),
        unit="deg/Myr",
        description="Velocity in Galactic latitude",
    )
    table["RA"] = Column(ra, unit="deg", description="Right ascension")
    table["DEC"] = Column(dec, unit="deg", description="Declination")

    try:
        luminosity = table["luminosity"]
        flux = astrometry.luminosity_to_flux(luminosity, distance)
        table["flux"] = Column(flux.value, unit=flux.unit, description="Source flux")
    except KeyError:
        pass

    try:
        extension = table["extension"]
        angular_extension = np.degrees(np.arctan(extension / distance))
        table["angular_extension"] = Column(
            angular_extension,
            unit="deg",
            description="Source angular radius (i.e. half-diameter)",
        )
    except KeyError:
        pass

    return table

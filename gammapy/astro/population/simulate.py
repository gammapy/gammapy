# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simulate source catalogs."""
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Column, Table
from astropy.units import Quantity
from gammapy.astro.source import PWN, SNR, Pulsar, SNRTrueloveMcKee
from gammapy.utils import coordinates as astrometry
from gammapy.utils.random import (
    draw,
    get_random_state,
    pdf,
    sample_sphere,
    sample_sphere_distance,
)
from .spatial import (
    RMAX,
    RMIN,
    ZMAX,
    ZMIN,
    Exponential,
    FaucherSpiral,
    radial_distributions,
)
from .velocity import VMAX, VMIN, velocity_distributions

__all__ = [
    "add_observed_parameters",
    "add_pulsar_parameters",
    "add_pwn_parameters",
    "add_snr_parameters",
    "make_base_catalog_galactic",
    "make_catalog_random_positions_cube",
    "make_catalog_random_positions_sphere",
]


def make_catalog_random_positions_cube(
    size=100, dimension=3, distance_max="1 pc", random_state="random-seed"
):
    """Make a catalog of sources randomly distributed on a line, square or cube.

    This can be used to study basic source population distribution effects,
    e.g. what the distance distribution looks like, or for a given luminosity
    function what the resulting flux distributions are for different spatial
    configurations.

    Parameters
    ----------
    size : int
        Number of sources
    dimension : {1, 2, 3}
        Number of dimensions
    distance_max : `~astropy.units.Quantity`
        Maximum distance
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.

    Returns
    -------
    table : `~astropy.table.Table`
        Table with 3D position cartesian coordinates.
        Columns: x (pc), y (pc), z (pc)
    """
    distance_max = Quantity(distance_max).to_value("pc")
    random_state = get_random_state(random_state)

    # Generate positions 1D, 2D, or 3D
    if dimension == 1:
        x = random_state.uniform(-distance_max, distance_max, size)
        y, z = 0, 0
    elif dimension == 2:
        x = random_state.uniform(-distance_max, distance_max, size)
        y = random_state.uniform(-distance_max, distance_max, size)
        z = 0
    elif dimension == 3:
        x = random_state.uniform(-distance_max, distance_max, size)
        y = random_state.uniform(-distance_max, distance_max, size)
        z = random_state.uniform(-distance_max, distance_max, size)
    else:
        raise ValueError(f"Invalid dimension: {dimension}")

    table = Table()
    table["x"] = Column(x, unit="pc", description="Cartesian coordinate")
    table["y"] = Column(y, unit="pc", description="Cartesian coordinate")
    table["z"] = Column(z, unit="pc", description="Cartesian coordinate")

    return table


def make_catalog_random_positions_sphere(
    size=100, distance_min="0 pc", distance_max="1 pc", random_state="random-seed"
):
    """Sample random source locations in a sphere.

    This can be used to generate an isotropic source population
    in a sphere, e.g. to represent extra-galactic sources.

    Parameters
    ----------
    size : int
        Number of sources
    distance_min, distance_max : `~astropy.units.Quantity`
        Minimum and maximum distance
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.

    Returns
    -------
    catalog : `~astropy.table.Table`
        Table with 3D position spherical coordinates.
        Columns: lon (deg), lat (deg), distance(pc)
    """
    distance_min = Quantity(distance_min).to_value("pc")
    distance_max = Quantity(distance_max).to_value("pc")
    random_state = get_random_state(random_state)

    lon, lat = sample_sphere(size, random_state=random_state)
    distance = sample_sphere_distance(distance_min, distance_max, size, random_state)

    table = Table()

    table["lon"] = Column(lon, unit="rad", description="Spherical coordinate")
    table["lat"] = Column(lat, unit="rad", description="Spherical coordinate")
    table["distance"] = Column(distance, unit="pc", description="Spherical coordinate")

    return table


def make_base_catalog_galactic(
    n_sources,
    rad_dis="YK04",
    vel_dis="H05",
    max_age="1e6 yr",
    spiralarms=True,
    n_ISM="1 cm-3",
    random_state="random-seed",
):
    """Make a catalog of Galactic sources, with basic source parameters.

    Choose a radial distribution, a velocity distribution, the number
    of pulsars n_pulsars, the maximal age max_age[years] and the fraction
    of the individual morphtypes. There's an option spiralarms. If set on
    True a spiralarm modeling after Faucher&Kaspi is included.

    max_age and n_sources effectively correspond to s SN rate:
    SN_rate = n_sources / max_age

    Parameters
    ----------
    n_sources : int
        Number of sources to simulate
    rad_dis : callable
        Radial surface density distribution of sources
    vel_dis : callable
        Proper motion velocity distribution of sources
    max_age : `~astropy.units.Quantity`
        Maximal age of the source
    spiralarms : bool
        Include a spiralarm model in the catalog?
    n_ISM : `~astropy.units.Quantity`
        Density of the interstellar medium
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.

    Returns
    -------
    table : `~astropy.table.Table`
        Catalog of simulated source positions and proper velocities
    """
    max_age = Quantity(max_age).to_value("yr")
    n_ISM = Quantity(n_ISM).to("cm-3")
    random_state = get_random_state(random_state)

    if isinstance(rad_dis, str):
        rad_dis = radial_distributions[rad_dis]

    if isinstance(vel_dis, str):
        vel_dis = velocity_distributions[vel_dis]

    # Draw random values for the age
    age = random_state.uniform(0, max_age, n_sources)
    age = Quantity(age, "yr")

    # Draw spatial distribution
    r = draw(
        RMIN.to_value("kpc"),
        RMAX.to_value("kpc"),
        n_sources,
        pdf(rad_dis()),
        random_state=random_state,
    )
    r = Quantity(r, "kpc")

    if spiralarms:
        r, theta, spiralarm = FaucherSpiral()(r, random_state=random_state)
    else:
        theta = Quantity(random_state.uniform(0, 2 * np.pi, n_sources), "rad")
        spiralarm = None

    x, y = astrometry.cartesian(r, theta)

    z = draw(
        ZMIN.to_value("kpc"),
        ZMAX.to_value("kpc"),
        n_sources,
        Exponential(),
        random_state=random_state,
    )
    z = Quantity(z, "kpc")

    # Draw values from velocity distribution
    v = draw(
        VMIN.to_value("km/s"),
        VMAX.to_value("km/s"),
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

    table = Table()
    table["age"] = Column(age, unit="yr", description="Age of the source")
    table["n_ISM"] = Column(n_ISM, description="Interstellar medium density")
    if spiralarms:
        table["spiralarm"] = Column(spiralarm, description="Which spiralarm?")

    table["x_birth"] = Column(x, description="Galactocentric x coordinate at birth")
    table["y_birth"] = Column(y, description="Galactocentric y coordinate at birth")
    table["z_birth"] = Column(z, description="Galactocentric z coordinate at birth")

    table["x"] = Column(x_moved.to("kpc"), description="Galactocentric x coordinate")
    table["y"] = Column(y_moved.to("kpc"), description="Galactocentric y coordinate")
    table["z"] = Column(z_moved.to("kpc"), description="Galactocentric z coordinate")

    table["vx"] = Column(vx, description="Galactocentric velocity in x direction")
    table["vy"] = Column(vy, description="Galactocentric velocity in y direction")
    table["vz"] = Column(vz, description="Galactocentric velocity in z direction")
    table["v_abs"] = Column(v, description="Galactocentric velocity (absolute)")

    return table


def add_snr_parameters(table):
    """Add SNR parameters to the table.

    TODO: document
    """
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
    table["E_SN"] = Column(E_SN, description="SNR kinetic energy")
    table["r_out"] = Column(r_out.to("pc"), description="SNR outer radius")
    table["r_in"] = Column(r_in.to("pc"), description="SNR inner radius")
    table["L_SNR"] = Column(L_SNR, description="SNR photon rate above 1 TeV")
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

    log10_b_psr = random_state.normal(B_mean, B_stdv, len(table))
    b_psr = Quantity(10**log10_b_psr, "G")

    # Compute pulsar parameters
    psr = Pulsar(p0_birth, b_psr)
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
    table["B_PSR"] = Column(
        b_psr, unit="Gauss", description="Pulsar magnetic field at the poles"
    )
    return table


def add_pwn_parameters(table):
    """Add PWN parameters to the table.

    TODO: document
    """
    # Some of the computations (specifically `pwn.radius`) aren't vectorised
    # across all parameters; so here we loop over source parameters explicitly

    results = []

    for idx in range(len(table)):
        age = table["age"].quantity[idx]
        E_SN = table["E_SN"].quantity[idx]
        n_ISM = table["n_ISM"].quantity[idx]
        P0_birth = table["P0_birth"].quantity[idx]
        b_psr = table["B_PSR"].quantity[idx]

        # Compute properties
        pulsar = Pulsar(P0_birth, b_psr)
        snr = SNRTrueloveMcKee(e_sn=E_SN, n_ISM=n_ISM)
        pwn = PWN(pulsar, snr)
        r_out_pwn = pwn.radius(age).to_value("pc")
        results.append(dict(r_out_pwn=r_out_pwn))

    # Add columns to table
    table["r_out_PWN"] = Column(
        [_["r_out_pwn"] for _ in results], unit="pc", description="PWN outer radius"
    )
    return table


def add_observed_parameters(table, obs_pos=None):
    """Add observable parameters (such as sky position or distance).

    Input table columns: x, y, z, extension, luminosity

    Output table columns: distance, glon, glat, flux, angular_extension

    Position of observer in cartesian coordinates.
    Center of galaxy as origin, x-axis goes through sun.

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
    obs_pos = obs_pos or [astrometry.D_SUN_TO_GALACTIC_CENTER, 0, 0]

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
        flux = luminosity / (4 * np.pi * distance**2)
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

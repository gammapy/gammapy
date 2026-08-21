# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
import pytest

from gammapy.astro.darkmatter import (
    DarkMatterSpectralModel,
    JFactory,
    profiles,
    add_factor_prior,
)
from gammapy.maps import WcsGeom
from gammapy.utils.testing import assert_quantity_allclose, requires_data


@pytest.fixture(scope="session")
def geom():
    return WcsGeom.create(binsz=0.5, npix=10)


@pytest.fixture(scope="session")
def jfact_annihilation(geom):
    jfactory = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=8.33 * u.kpc,
        rmax=1 * u.kpc,
    )
    return jfactory.compute_jfactor()


@pytest.fixture(scope="session")
def jfact_decay(geom):
    jfactory = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=8.33 * u.kpc,
        annihilation=False,
        rmax=1 * u.kpc,
    )
    return jfactory.compute_jfactor()


@pytest.fixture
def dm_decay_model():
    return DarkMatterSpectralModel(
        mDM=5000 * u.Unit("GeV"),
        channel="b",
        factor=3.41e19 * u.Unit("GeV cm-2"),
        annihilation=False,
    )


def test_compute_differential_jfactor_large_separation():
    geom = WcsGeom.create(skydir=(0, 0), width=(120, 2), binsz=1, frame="galactic")
    assert geom.separation(geom.center_skydir).deg.max() > 45

    jfactory = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=8.33 * u.kpc,
        rmax=1 * u.kpc,
    )

    jfactor = jfactory.compute_differential_jfactor(ndecade=100)

    assert jfactor.shape == geom.data_shape
    assert np.all(np.isfinite(jfactor.value))


def test_compute_differential_jfactor_outside_halo_no_intersection():
    geom = WcsGeom.create(skydir=(0, 0), width=(120, 2), binsz=1, frame="galactic")
    separation = geom.separation(geom.center_skydir)

    jfactory = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=8.33 * u.kpc,
        rmax=1 * u.kpc,
    )

    jfactor = jfactory.compute_differential_jfactor(ndecade=100)

    max_intersection_angle = (
        np.arcsin((jfactory.rmax / jfactory.distance).to_value("")) * u.rad
    )

    assert np.all(jfactor[separation >= max_intersection_angle] == 0)


@pytest.mark.parametrize(
    ("name", "distance", "rmax", "separation"),
    [
        ("inside_toward", 1, 2, 30),
        ("inside_perpendicular", 1, 2, 90),
        ("inside_away", 1, 2, 120),
        ("boundary_toward", 2, 2, 30),
        ("boundary_away", 2, 2, 120),
        ("outside_intersects", 10, 2, 5),
        ("outside_misses", 10, 2, 30),
        ("outside_away", 10, 2, 120),
    ],
)
def test_integrate_los_geometric_path_length(
    geom,
    name,
    distance,
    rmax,
    separation,
):
    density = 1 * u.GeV / u.cm**3

    def constant_profile(radius):
        return np.ones(np.shape(radius)) * density

    jfactory = JFactory(
        geom=geom,
        profile=constant_profile,
        distance=distance * u.kpc,
        rmax=rmax * u.kpc,
    )

    theta = np.deg2rad(separation)
    impact = jfactory.distance * np.sin(theta)

    actual = jfactory._integrate_los(
        impact=impact,
        separation=theta,
        ndecade=10000,
    )

    discriminant = jfactory.rmax**2 - impact**2

    if discriminant <= 0 * u.kpc**2:
        path_length = 0 * u.kpc
    else:
        root = np.sqrt(discriminant)
        los_center = jfactory.distance * np.cos(theta)
        los_min = los_center - root
        los_max = los_center + root

        if np.isclose(los_max.to_value(u.kpc), 0, atol=1e-12) or los_max < 0 * u.kpc:
            path_length = 0 * u.kpc
        elif los_min <= 0 * u.kpc:
            path_length = los_max
        else:
            path_length = los_max - los_min

    expected = density**2 * path_length

    assert_quantity_allclose(actual, expected, rtol=1e-5)


def test_integrate_los_branch_zero_impact_positive_radius():
    geom = WcsGeom.create(binsz=1, npix=2)
    profile = profiles.NFWProfile()
    jfactory = JFactory(
        geom=geom,
        profile=profile,
        distance=8.33 * u.kpc,
        rmax=1 * u.kpc,
    )

    radius_min = 1 * u.kpc
    radius_max = 4 * u.kpc

    actual = jfactory._integrate_los_branch(
        0 * u.kpc, radius_min, radius_max, ndecade=100
    )

    desired = profile.integral(
        rmin=radius_min,
        rmax=radius_max,
        separation=0,
        ndecade=100,
        squared=True,
        distance=8.33 * u.kpc,
    )

    assert_quantity_allclose(actual, desired)


@requires_data()
def test_dmfluxmap_annihilation(jfact_annihilation):
    energy_min = 0.1 * u.TeV
    energy_max = 10 * u.TeV
    massDM = 1 * u.TeV
    channel = "W"

    total_jfact = u.Quantity(
        float(jfact_annihilation.mean().value), unit=jfact_annihilation.unit
    )

    diff_flux = DarkMatterSpectralModel(mDM=massDM, channel=channel, factor=total_jfact)
    int_flux = (
        diff_flux.integral(energy_min=energy_min, energy_max=energy_max)
        * jfact_annihilation
        / total_jfact
    ).to("cm-2 s-1")
    actual = int_flux[5, 5]
    desired = 5.84534173e-12 / u.cm**2 / u.s

    assert_quantity_allclose(actual, desired, rtol=1e-3)


@requires_data()
def test_dmfluxmap_decay(jfact_decay):
    energy_min = 0.1 * u.TeV
    energy_max = 10 * u.TeV
    massDM = 1 * u.TeV
    channel = "W"

    diff_flux = DarkMatterSpectralModel(mDM=massDM, channel=channel, annihilation=False)
    int_flux = (
        jfact_decay
        * diff_flux.integral(energy_min=energy_min, energy_max=energy_max)
        / diff_flux.factor
    ).to("cm-2 s-1")
    actual = int_flux[5, 5]
    desired = 1.09754e-3 / u.cm**2 / u.s
    assert_quantity_allclose(actual, desired, rtol=1e-3)


@requires_data()
def test_prior_attached(dm_decay_model):
    """The prior should be a GaussianPrior with the given sigma,
    centered on mu=1 by default (i.e. the nominal factor value)."""
    add_factor_prior(dm_decay_model, sigma=0.2)

    prior = dm_decay_model.scale.prior
    assert prior is not None
    assert prior.sigma.value == pytest.approx(0.2 * np.log(10))
    assert prior.mu.value == pytest.approx(1.0)


@requires_data()
def test_custom_mu(dm_decay_model):
    """A custom `mu` should be respected instead of the default 1.0."""
    add_factor_prior(dm_decay_model, sigma=0.15, mu=0.5)

    prior = dm_decay_model.scale.prior
    assert prior.mu.value == pytest.approx(0.5)
    assert prior.sigma.value == pytest.approx(0.15 * np.log(10))


@requires_data()
def test_jfactor_unaffected(dm_decay_model):
    """The nominal factor attribute itself should remain untouched;
    only `scale` should carry the nuisance treatment."""
    factor_before = dm_decay_model.factor

    add_factor_prior(dm_decay_model, sigma=0.2)

    assert dm_decay_model.factor == factor_before


@requires_data()
def test_returns_model(dm_decay_model):
    """The function should return the same model instance (for chaining)."""
    returned = add_factor_prior(dm_decay_model, sigma=0.2)
    assert returned is dm_decay_model


@requires_data()
def test_flux_scale_degeneracy_regression(dm_decay_model):
    """Evaluating the model at scale=1 must reproduce the
    flux computed with the nominal factor, i.e. the prior on `scale`
    does not change the model's evaluate() behaviour by itself."""
    energy = 100 * u.GeV
    flux_before = dm_decay_model(energy)

    add_factor_prior(dm_decay_model, sigma=0.2)
    # attaching the prior alone shouldn't move scale's current value
    flux_after = dm_decay_model(energy)

    assert u.allclose(flux_before, flux_after)

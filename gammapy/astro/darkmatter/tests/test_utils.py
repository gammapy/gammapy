# Licensed under a 3-clause BSD style license - see LICENSE.rst
from unittest.mock import patch
import numpy as np
import html
import astropy.units as u
import pytest
from numpy.testing import assert_allclose

from gammapy.astro.darkmatter import (
    DarkMatterAnnihilationSpectralModel,
    DarkMatterDecaySpectralModel,
    JFactory,
    profiles,
    add_factor_prior,
)
from gammapy.maps import WcsGeom
from gammapy.modeling import Parameter
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
    return DarkMatterDecaySpectralModel(
        mass=5000 * u.Unit("GeV"),
        channel="b",
        jfactor=3.41e19 * u.Unit("GeV cm-2"),
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

    diff_flux = DarkMatterAnnihilationSpectralModel(
        mass=massDM, channel=channel, jfactor=total_jfact
    )
    int_flux = (
        diff_flux.integral(energy_min=energy_min, energy_max=energy_max)
        * jfact_annihilation
        / total_jfact
    ).to("cm-2 s-1")
    actual = int_flux[5, 5]
    desired = 5.902332e-12 / u.cm**2 / u.s

    assert_quantity_allclose(actual, desired, rtol=1e-3)


@requires_data()
def test_dmfluxmap_decay(jfact_decay):
    energy_min = 0.1 * u.TeV
    energy_max = 10 * u.TeV
    massDM = 1 * u.TeV
    channel = "W"

    diff_flux = DarkMatterDecaySpectralModel(mass=massDM, channel=channel)
    int_flux = (
        jfact_decay
        * diff_flux.integral(energy_min=energy_min, energy_max=energy_max)
        / diff_flux.jfactor
    ).to("cm-2 s-1")
    actual = int_flux[5, 5]
    desired = 1.277e-3 / u.cm**2 / u.s
    assert_quantity_allclose(actual, desired, rtol=1e-3)


@requires_data()
def test_prior_attached(dm_decay_model):
    """The prior should be a GaussianPrior with the given sigma,
    centered on mu=1 by default (i.e. the nominal factor value)."""
    add_factor_prior(dm_decay_model, sigma=0.2)

    prior = dm_decay_model.scale.prior
    assert prior is not None
    assert prior.sigma.value == pytest.approx(0.2)
    assert prior.mu.value == pytest.approx(1.0)


@requires_data()
def test_custom_mu(dm_decay_model):
    """A custom `mu` should be respected instead of the default 1.0."""
    add_factor_prior(dm_decay_model, sigma=0.15, mu=0.5)

    prior = dm_decay_model.scale.prior
    assert prior.mu.value == pytest.approx(0.5)
    assert prior.sigma.value == pytest.approx(0.15)


@requires_data()
def test_jfactor_unaffected(dm_decay_model):
    """The nominal factor attribute itself should remain untouched;
    only `scale` should carry the nuisance treatment."""
    jfactor_before = dm_decay_model.jfactor

    add_factor_prior(dm_decay_model, sigma=0.2)

    assert dm_decay_model.jfactor == jfactor_before


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

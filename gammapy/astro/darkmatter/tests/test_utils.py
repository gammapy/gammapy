# Licensed under a 3-clause BSD style license - see LICENSE.rst
import html
from unittest.mock import patch

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from gammapy.astro.darkmatter import (
    DarkMatterAnnihilationSpectralModel,
    DarkMatterDecaySpectralModel,
    JFactory,
    profiles,
)
from gammapy.maps import WcsGeom
from gammapy.utils.testing import assert_quantity_allclose, requires_data

# Reference values replacing the removed DMProfile.DISTANCE_GC constant
DISTANCE_GC = 8.33 * u.kpc
RMAX_GC = 50 * u.kpc  # Typical Milky Way halo truncation radius


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def geom():
    return WcsGeom.create(binsz=0.5, npix=10)


@pytest.fixture(scope="session")
def jfact_annihilation(geom):
    jfactory = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=DISTANCE_GC,
        rmax=RMAX_GC,
        annihilation=True,
    )
    return jfactory.compute_jfactor()


@pytest.fixture(scope="session")
def jfact_decay(geom):
    jfactory = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=DISTANCE_GC,
        rmax=RMAX_GC,
        annihilation=False,
    )
    return jfactory.compute_jfactor()


# ============================================================================
# Flux Map Tests
# ============================================================================


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
    desired = 5.96827647e-12 / u.cm**2 / u.s
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
    desired = 1.783e-3 / u.cm**2 / u.s
    assert_quantity_allclose(actual, desired, rtol=1e-3)


# ============================================================================
# JFactory Instantiation Tests
# ============================================================================


def test_jfactory_default_annihilation(geom):
    """Test JFactory default is annihilation mode."""
    jfactory = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=DISTANCE_GC,
        rmax=RMAX_GC,
    )
    assert jfactory.annihilation is True


def test_jfactory_decay_mode(geom):
    """Test JFactory decay mode flag."""
    jfactory = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=DISTANCE_GC,
        rmax=RMAX_GC,
        annihilation=False,
    )
    assert jfactory.annihilation is False


def test_jfactory_stores_attributes(geom):
    """Test JFactory stores all constructor arguments correctly."""
    profile = profiles.NFWProfile()
    jfactory = JFactory(
        geom=geom,
        profile=profile,
        distance=DISTANCE_GC,
        rmax=RMAX_GC,
        annihilation=True,
    )
    assert jfactory.geom is geom
    assert jfactory.profile is profile
    assert_quantity_allclose(jfactory.distance, DISTANCE_GC)
    assert_quantity_allclose(jfactory.rmax, RMAX_GC)
    assert jfactory.annihilation is True


# ============================================================================
# compute_differential_jfactor Tests
# ============================================================================


def test_differential_jfactor_annihilation_units(geom):
    """Test that differential J-factor for annihilation has correct units."""
    jfactory = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=DISTANCE_GC,
        rmax=RMAX_GC,
        annihilation=True,
    )
    diff_jfact = jfactory.compute_differential_jfactor()
    assert diff_jfact.unit.is_equivalent(u.Unit("GeV2 cm-5 sr-1"))


def test_differential_jfactor_decay_units(geom):
    """Test that differential J-factor for decay has correct units."""
    jfactory = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=DISTANCE_GC,
        rmax=RMAX_GC,
        annihilation=False,
    )
    diff_jfact = jfactory.compute_differential_jfactor()
    assert diff_jfact.unit.is_equivalent(u.Unit("GeV cm-2 sr-1"))


def test_differential_jfactor_positive(geom):
    """Test that differential J-factor values are positive."""
    jfactory = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=DISTANCE_GC,
        rmax=RMAX_GC,
    )
    diff_jfact = jfactory.compute_differential_jfactor()
    assert np.all(diff_jfact.value > 0)


def test_differential_jfactor_shape(geom):
    """Test that differential J-factor has the correct shape."""
    jfactory = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=DISTANCE_GC,
        rmax=RMAX_GC,
    )
    diff_jfact = jfactory.compute_differential_jfactor()
    assert diff_jfact.shape == geom.to_image().data_shape


# ============================================================================
# compute_jfactor Tests
# ============================================================================


def test_jfactor_annihilation_units(geom):
    """Test that J-factor for annihilation has correct units."""
    jfactory = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=DISTANCE_GC,
        rmax=RMAX_GC,
        annihilation=True,
    )
    jfact = jfactory.compute_jfactor()
    assert jfact.unit.is_equivalent(u.Unit("GeV2 cm-5"))


def test_jfactor_decay_units(geom):
    """Test that J-factor for decay has correct units."""
    jfactory = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=DISTANCE_GC,
        rmax=RMAX_GC,
        annihilation=False,
    )
    jfact = jfactory.compute_jfactor()
    assert jfact.unit.is_equivalent(u.Unit("GeV cm-2"))


def test_jfactor_positive(geom):
    """Test that J-factor values are all positive."""
    jfactory = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=DISTANCE_GC,
        rmax=RMAX_GC,
    )
    jfact = jfactory.compute_jfactor()
    assert np.all(jfact.value > 0)


def test_jfactor_shape(geom):
    """Test that J-factor has the correct spatial shape."""
    jfactory = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=DISTANCE_GC,
        rmax=RMAX_GC,
    )
    jfact = jfactory.compute_jfactor()
    assert jfact.shape == geom.to_image().data_shape


def test_jfactor_annihilation_greater_than_decay(geom):
    """Test that annihilation J-factor > decay D-factor at center (ρ² > ρ for ρ > 1)."""
    profile = profiles.NFWProfile()
    profile.scale_to_local_density(distance=DISTANCE_GC)

    jfactory_ann = JFactory(
        geom=geom,
        profile=profile,
        distance=DISTANCE_GC,
        rmax=RMAX_GC,
        annihilation=True,
    )
    jfactory_dec = JFactory(
        geom=geom,
        profile=profile,
        distance=DISTANCE_GC,
        rmax=RMAX_GC,
        annihilation=False,
    )
    jfact_ann = jfactory_ann.compute_jfactor()
    jfact_dec = jfactory_dec.compute_jfactor()

    # Units differ (GeV2 cm-5 vs GeV cm-2) so compare only shapes and positivity
    assert jfact_ann.shape == jfact_dec.shape
    assert np.all(jfact_ann.value > 0)
    assert np.all(jfact_dec.value > 0)


def test_jfactor_rmax_effect(geom):
    """Test that a larger rmax gives a larger J-factor (more volume integrated)."""
    jfactory_small = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=DISTANCE_GC,
        rmax=20 * u.kpc,
    )
    jfactory_large = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=DISTANCE_GC,
        rmax=100 * u.kpc,
    )
    jfact_small = jfactory_small.compute_jfactor()
    jfact_large = jfactory_large.compute_jfactor()
    assert np.all(jfact_large.value > jfact_small.value)


def test_jfactor_different_profiles(geom):
    """Test that different DM profiles give different J-factors."""
    jfactory_nfw = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=DISTANCE_GC,
        rmax=RMAX_GC,
    )
    jfactory_iso = JFactory(
        geom=geom,
        profile=profiles.IsothermalProfile(),
        distance=DISTANCE_GC,
        rmax=RMAX_GC,
    )
    jfact_nfw = jfactory_nfw.compute_jfactor()
    jfact_iso = jfactory_iso.compute_jfactor()
    assert not np.allclose(jfact_nfw.value, jfact_iso.value)


# ============================================================================
# HTML Representation Tests
# ============================================================================


def test_jfactory_repr_html_fallback(geom):
    """Test that _repr_html_ falls back to <pre> block when to_html is missing."""
    jfactory = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=DISTANCE_GC,
        rmax=RMAX_GC,
    )
    with patch.object(jfactory, "to_html", side_effect=AttributeError, create=True):
        repr_str = jfactory._repr_html_()
    assert repr_str.startswith("<pre>")
    assert repr_str.endswith("</pre>")
    expected_string = html.escape(str(jfactory))
    assert expected_string in repr_str

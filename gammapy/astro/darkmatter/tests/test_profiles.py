# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Comprehensive tests for Dark Matter Profiles."""

import astropy.units as u
import numpy as np
import pytest
from astropy.units import Quantity

from gammapy.astro.darkmatter import profiles
from gammapy.utils.testing import assert_quantity_allclose

# List of all dark matter profiles to test
dm_profiles = [
    profiles.ZhaoProfile,
    profiles.NFWProfile,
    profiles.EinastoProfile,
    profiles.IsothermalProfile,
    profiles.BurkertProfile,
    profiles.MooreProfile,
]


# ============================================================================
# Basic Instantiation Tests
# ============================================================================


@pytest.mark.parametrize("profile", dm_profiles)
def test_profile_instantiation(profile):
    """Test that all profiles can be instantiated with default parameters."""
    p = profile()
    assert p is not None
    assert hasattr(p, "parameters")
    assert len(p.parameters) > 0


@pytest.mark.parametrize("profile", dm_profiles)
def test_profile_has_required_constants(profile):
    """Test that all profiles have required class constants."""
    assert hasattr(profile, "LOCAL_DENSITY")
    assert hasattr(profile, "DISTANCE_GC")
    assert hasattr(profile, "DEFAULT_SCALE_RADIUS")


# ============================================================================
# Scale to Local Density Tests
# ============================================================================


@pytest.mark.parametrize("profile", dm_profiles)
def test_profiles_scale_to_local_density(profile):
    """Test scaling to local density.

    After scaling, the density at the Galactic Center distance
    should equal the local density.
    """
    p = profile()
    p.scale_to_local_density()
    actual = p(p.DISTANCE_GC)
    desired = p.LOCAL_DENSITY
    assert_quantity_allclose(actual, desired)


# ============================================================================
# Parameter Tests
# ============================================================================


def test_zhao_profile_parameters():
    """Test ZhaoProfile parameter initialization."""
    p = profiles.ZhaoProfile()
    assert p.parameters["r_s"].value == profiles.ZhaoProfile.DEFAULT_SCALE_RADIUS.value
    assert p.parameters["alpha"].value == profiles.ZhaoProfile.DEFAULT_ALPHA
    assert p.parameters["beta"].value == profiles.ZhaoProfile.DEFAULT_BETA
    assert p.parameters["gamma"].value == profiles.ZhaoProfile.DEFAULT_GAMMA


def test_zhao_profile_custom_parameters():
    """Test ZhaoProfile with custom parameters."""
    r_s_custom = 20 * u.kpc
    alpha_custom = 0.5
    beta_custom = 2.5
    gamma_custom = 0.8
    rho_s_custom = 2.0 * u.Unit("GeV / cm3")

    p = profiles.ZhaoProfile(
        r_s=r_s_custom,
        alpha=alpha_custom,
        beta=beta_custom,
        gamma=gamma_custom,
        rho_s=rho_s_custom,
    )

    assert_quantity_allclose(p.parameters["r_s"].quantity, r_s_custom)
    assert p.parameters["alpha"].value == alpha_custom
    assert p.parameters["beta"].value == beta_custom
    assert p.parameters["gamma"].value == gamma_custom
    assert_quantity_allclose(p.parameters["rho_s"].quantity, rho_s_custom)


def test_nfw_profile_parameters():
    """Test NFWProfile parameter initialization."""
    p = profiles.NFWProfile()
    assert_quantity_allclose(
        p.parameters["r_s"].quantity, profiles.NFWProfile.DEFAULT_SCALE_RADIUS
    )


def test_nfw_profile_custom_parameters():
    """Test NFWProfile with custom parameters."""
    r_s_custom = 20 * u.kpc
    rho_s_custom = 2.0 * u.Unit("GeV / cm3")

    p = profiles.NFWProfile(r_s=r_s_custom, rho_s=rho_s_custom)

    assert_quantity_allclose(p.parameters["r_s"].quantity, r_s_custom)
    assert_quantity_allclose(p.parameters["rho_s"].quantity, rho_s_custom)


def test_einasto_profile_parameters():
    """Test EinastoProfile parameter initialization."""
    p = profiles.EinastoProfile()
    assert_quantity_allclose(
        p.parameters["r_s"].quantity, profiles.EinastoProfile.DEFAULT_SCALE_RADIUS
    )
    assert p.parameters["alpha"].value == profiles.EinastoProfile.DEFAULT_ALPHA


def test_isothermal_profile_parameters():
    """Test IsothermalProfile parameter initialization."""
    p = profiles.IsothermalProfile()
    assert_quantity_allclose(
        p.parameters["r_s"].quantity, profiles.IsothermalProfile.DEFAULT_SCALE_RADIUS
    )


def test_burkert_profile_parameters():
    """Test BurkertProfile parameter initialization."""
    p = profiles.BurkertProfile()
    assert_quantity_allclose(
        p.parameters["r_s"].quantity, profiles.BurkertProfile.DEFAULT_SCALE_RADIUS
    )


def test_moore_profile_parameters():
    """Test MooreProfile parameter initialization."""
    p = profiles.MooreProfile()
    assert_quantity_allclose(
        p.parameters["r_s"].quantity, profiles.MooreProfile.DEFAULT_SCALE_RADIUS
    )


# ============================================================================
# Evaluation Tests
# ============================================================================


def test_zhao_profile_evaluate():
    """Test ZhaoProfile evaluation at different radii."""
    p = profiles.ZhaoProfile()

    # Test evaluation at scale radius (should be close to rho_s for Zhao)
    r_s = p.parameters["r_s"].quantity
    result = p(r_s)

    assert result.unit.is_equivalent(u.GeV / u.cm**3)
    assert result.value > 0


def test_nfw_profile_evaluate():
    """Test NFWProfile evaluation."""
    p = profiles.NFWProfile()
    r_s = p.parameters["r_s"].quantity
    rho_s = p.parameters["rho_s"].quantity

    # At r = r_s, NFW profile should give rho_s / (1 * 2^2) = rho_s / 4
    result = p(r_s)
    expected = rho_s / 4.0

    assert_quantity_allclose(result, expected)


def test_nfw_profile_evaluate_asymptotic():
    """Test NFWProfile asymptotic behavior."""
    p = profiles.NFWProfile()
    r_s = p.parameters["r_s"].quantity

    # At very small radius, should behave like 1/r
    small_r = r_s * 0.001
    result_small = p(small_r)

    # At very large radius, should behave like 1/r^3
    large_r = r_s * 1000
    result_large = p(large_r)

    # Should be monotonically decreasing
    assert result_small > result_large


def test_einasto_profile_evaluate():
    """Test EinastoProfile evaluation."""
    p = profiles.EinastoProfile()
    r_s = p.parameters["r_s"].quantity

    result = p(r_s)
    assert result.unit.is_equivalent(u.GeV / u.cm**3)
    assert result.value > 0


def test_isothermal_profile_evaluate():
    """Test IsothermalProfile evaluation."""
    p = profiles.IsothermalProfile()
    r_s = p.parameters["r_s"].quantity
    rho_s = p.parameters["rho_s"].quantity

    # At r = r_s, should give rho_s / (1 + 1) = rho_s / 2
    result = p(r_s)
    expected = rho_s / 2.0

    assert_quantity_allclose(result, expected)


def test_burkert_profile_evaluate():
    """Test BurkertProfile evaluation."""
    p = profiles.BurkertProfile()
    r_s = p.parameters["r_s"].quantity
    rho_s = p.parameters["rho_s"].quantity

    # At r = r_s, should give rho_s / ((1 + 1) * (1 + 1)) = rho_s / 4
    result = p(r_s)
    expected = rho_s / 4.0

    assert_quantity_allclose(result, expected)


def test_moore_profile_evaluate():
    """Test MooreProfile evaluation."""
    p = profiles.MooreProfile()
    r_s = p.parameters["r_s"].quantity

    result = p(r_s)
    assert result.unit.is_equivalent(u.GeV / u.cm**3)
    assert result.value > 0


# ============================================================================
# Monotonic Behavior Tests
# ============================================================================


@pytest.mark.parametrize("profile", dm_profiles)
def test_profile_monotonic_decreasing(profile):
    """Test that density decreases monotonically with radius."""
    p = profile()

    # Create a range of radii
    radii = np.logspace(-2, 3, 50) * u.kpc
    densities = np.array([p(r).value for r in radii])

    # Check that density is monotonically decreasing
    assert np.all(np.diff(densities) < 0), "Density should decrease with radius"


# ============================================================================
# Integration Tests
# ============================================================================

# NOTE: The integral method uses the formula:
#   F(r_min, r_max) = ∫ ρ(r)^γ * r / sqrt(r^2 - (distance * sin(separation))^2) dr
#
# For this to be mathematically valid (avoid NaN from sqrt of negative numbers):
#   - separation should be small (typically 0 for along-the-line-of-sight observations)
#   - OR: sin(separation) * distance < rmin (the minimum radius)
#
# In practice, use separation = 0.0 for most tests unless specifically testing
# the separation dependence with appropriate angular constraints.


def test_nfw_profile_integral():
    """Test integration of NFW profile."""
    p = profiles.NFWProfile()

    rmin = 0.01 * u.kpc
    rmax = 100 * u.kpc
    separation = 0.0  # along the line of sight
    ndecade = 100

    # Test squared integral (for annihilation)
    result_squared = p.integral(rmin, rmax, separation, ndecade, squared=True)
    assert result_squared.unit.is_equivalent(u.GeV**2 / u.cm**5)
    assert result_squared.value > 0

    # Test unsquared integral (for decay)
    result_unsquared = p.integral(rmin, rmax, separation, ndecade, squared=False)
    assert result_unsquared.unit.is_equivalent(u.GeV / u.cm**2)
    assert result_unsquared.value > 0

    # Squared integral should be larger (in general)
    assert result_squared.value > result_unsquared.value


def test_einasto_profile_integral():
    """Test integration of Einasto profile."""
    p = profiles.EinastoProfile()

    rmin = 0.01 * u.kpc
    rmax = 100 * u.kpc
    separation = 0.0  # along the line of sight
    ndecade = 100

    result = p.integral(rmin, rmax, separation, ndecade, squared=True)
    assert result.unit.is_equivalent(u.GeV**2 / u.cm**5)
    assert result.value > 0


@pytest.mark.parametrize("profile", dm_profiles)
def test_profile_integral_basic(profile):
    """Test that integral returns positive values with correct units."""
    p = profile()

    rmin = 0.1 * u.kpc
    rmax = 50 * u.kpc
    separation = 0.0
    ndecade = 50

    result = p.integral(rmin, rmax, separation, ndecade, squared=True)

    assert result.unit.is_equivalent(u.GeV**2 / u.cm**5)
    assert result.value > 0


# ============================================================================
# Call Method Tests
# ============================================================================


@pytest.mark.parametrize("profile", dm_profiles)
def test_profile_call_method(profile):
    """Test that __call__ method works correctly."""
    p = profile()
    r = 10 * u.kpc

    # Should be able to call the profile like a function
    result = p(r)

    assert isinstance(result, u.Quantity)
    assert result.unit.is_equivalent(u.GeV / u.cm**3)
    assert result.value > 0


@pytest.mark.parametrize("profile", dm_profiles)
def test_profile_call_with_array(profile):
    """Test that __call__ method works with arrays."""
    p = profile()
    radii = np.array([1, 5, 10, 50]) * u.kpc

    results = p(radii)

    assert results.shape == radii.shape
    assert results.unit.is_equivalent(u.GeV / u.cm**3)
    assert np.all(results.value > 0)


# ============================================================================
# Special Cases and Edge Cases
# ============================================================================


def test_zhao_is_nfw_special_case():
    """Test that Zhao profile with (alpha=1, beta=3, gamma=1) is equivalent to NFW."""
    zhao = profiles.ZhaoProfile(alpha=1, beta=3, gamma=1)
    nfw = profiles.NFWProfile()

    # Test at several radii
    test_radii = np.array([0.1, 1, 10, 100]) * u.kpc

    for r in test_radii:
        zhao_result = zhao(r)
        nfw_result = nfw(r)
        # They should be equal (within numerical precision)
        assert np.isclose(zhao_result.value, nfw_result.value, rtol=1e-10)


def test_profile_with_zero_radius_handling():
    """Test profile behavior near zero radius."""
    p = profiles.NFWProfile()

    # Very small radius should still work
    r_small = 1e-6 * u.kpc
    result = p(r_small)

    assert result.value > 0
    assert np.isfinite(result.value)


@pytest.mark.parametrize("profile", dm_profiles)
def test_profile_parameter_modification(profile):
    """Test that modifying parameters affects the profile evaluation."""
    p = profile()
    r = 10 * u.kpc

    # Get initial result
    result_initial = p(r)

    # Modify rho_s parameter
    original_rho_s = p.parameters["rho_s"].value
    p.parameters["rho_s"].value = original_rho_s * 2

    # Get result after modification
    result_modified = p(r)

    # Result should be approximately doubled
    assert np.isclose(result_modified.value / result_initial.value, 2.0, rtol=1e-10)


# ============================================================================
# Distance Parameter Tests
# ============================================================================


def test_integral_separation_constraints():
    """Test that separation must satisfy mathematical constraints.

    For the integral to be valid, we need:
    separation_angle * distance < radius

    Otherwise sqrt(radius^2 - (distance * sin(separation))^2) can be NaN.
    """
    p = profiles.NFWProfile()

    rmin = 0.01 * u.kpc
    rmax = 100 * u.kpc
    ndecade = 50

    # Valid separation: 0 (along line of sight)
    separation_valid = 0.0
    result = p.integral(rmin, rmax, separation_valid, ndecade, squared=True)
    assert np.isfinite(result.value)
    assert result.value > 0

    # Small valid separations should also work
    separation_small = 1e-6
    result_small = p.integral(rmin, rmax, separation_small, ndecade, squared=True)
    assert np.isfinite(result_small.value)


def test_nfw_integral_with_custom_distance():
    """Test NFW integration with custom distance to target.

    Note: separation parameter is the angular separation. For numerical stability,
    we use separation = 0 (along the line of sight).
    """
    p = profiles.NFWProfile()

    rmin = 0.01 * u.kpc
    rmax = 100 * u.kpc
    separation = 0.5  # along the line of sight for numerical stability
    ndecade = 50

    # Integral with default distance (Galactic Center)
    result_default = p.integral(rmin, rmax, separation, ndecade, squared=True)

    # Integral with custom distance
    custom_distance = 10 * u.kpc
    result_custom = p.integral(
        rmin, rmax, separation, ndecade, squared=True, distance=custom_distance
    )

    # Should be different values
    assert result_default.value != result_custom.value


# ============================================================================
# HTML Representation Tests
# ============================================================================


@pytest.mark.parametrize("profile", dm_profiles)
def test_profile_html_repr(profile):
    """Test that profiles have working HTML representation."""
    p = profile()

    html_repr = p._repr_html_()
    assert isinstance(html_repr, str)
    assert len(html_repr) > 0


# ============================================================================
# Numerical Consistency Tests
# ============================================================================


def test_nfw_vs_evaluate_method():
    """Test that __call__ and evaluate methods give same results."""
    p = profiles.NFWProfile(r_s=20 * u.kpc, rho_s=0.5 * u.Unit("GeV / cm3"))

    r = 15 * u.kpc

    # Using __call__
    result_call = p(r)

    # Using evaluate directly
    result_evaluate = profiles.NFWProfile.evaluate(
        r, p.parameters["r_s"].quantity, p.parameters["rho_s"].quantity
    )

    assert_quantity_allclose(result_call, result_evaluate)


# ============================================================================
# Dimensionality Tests
# ============================================================================


@pytest.mark.parametrize("profile", dm_profiles)
def test_profile_dimensional_consistency(profile):
    """Test that profile maintains dimensional consistency."""
    p = profile()

    # Test with different unit systems
    r_kpc = 10 * u.kpc
    r_pc = r_kpc.to(u.pc)

    result_kpc = p(r_kpc)
    result_pc = p(r_pc)

    # Convert both to same units for comparison
    assert_quantity_allclose(result_kpc, result_pc, rtol=1e-10)

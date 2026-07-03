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

# Reference distance (solar neighborhood to GC) for tests requiring a normalization distance
DISTANCE_REF = 8.33 * u.kpc
# Reference distance for integral tests
DISTANCE_INTEGRAL = 8.5 * u.kpc

# Basic Instantiation Tests


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
    assert hasattr(profile, "DEFAULT_SCALE_RADIUS")


# Distance Validation Tests (integral / integrate_spectrum_separation)


@pytest.mark.parametrize("profile", dm_profiles)
def test_integral_distance_none_raises(profile):
    """Test that integral() raises TypeError when distance is not provided."""
    p = profile()

    rmin = 0.1 * u.kpc
    rmax = 50 * u.kpc
    separation = 0.0
    ndecade = 50

    with pytest.raises(TypeError, match="missing required argument: 'distance'"):
        p.integral(rmin, rmax, separation, ndecade)


@pytest.mark.parametrize("profile", dm_profiles)
def test_integral_distance_as_bool_raises(profile):
    """Test that integral() raises TypeError when a bool is passed in the
    position of distance (old positional argument order, pre-v2.2)."""
    p = profile()

    rmin = 0.1 * u.kpc
    rmax = 50 * u.kpc
    separation = 0.0
    ndecade = 50

    with pytest.raises(TypeError, match="was added before 'squared'"):
        p.integral(rmin, rmax, separation, ndecade, 10, True)


def test_integrate_spectrum_separation_distance_none_raises():
    """Test that integrate_spectrum_separation() raises TypeError when
    distance is not provided."""
    p = profiles.NFWProfile()

    rmin = 0.1 * u.kpc
    rmax = 50 * u.kpc
    separation = 0.0
    ndecade = 50

    with pytest.raises(TypeError, match="missing required argument: 'distance'"):
        p.integrate_spectrum_separation(
            p._eval_substitution, rmin, rmax, separation, ndecade
        )


def test_integrate_spectrum_separation_distance_as_bool_raises():
    """Test that integrate_spectrum_separation() raises TypeError when a
    bool is passed in the position of distance."""
    p = profiles.NFWProfile()

    rmin = 0.1 * u.kpc
    rmax = 50 * u.kpc
    separation = 0.0
    ndecade = 50

    with pytest.raises(TypeError, match="was added before 'squared'"):
        p.integrate_spectrum_separation(
            p._eval_substitution, rmin, rmax, separation, ndecade, 10, True
        )


@pytest.mark.parametrize("profile", dm_profiles)
def test_integral_distance_valid_still_works(profile):
    """Sanity check: passing distance correctly should not raise and
    should return a finite, positive result."""
    p = profile()

    rmin = 0.1 * u.kpc
    rmax = 50 * u.kpc
    separation = 0.0
    ndecade = 50

    result = p.integral(
        rmin, rmax, separation, ndecade, distance=DISTANCE_INTEGRAL, squared=True
    )

    assert np.isfinite(result.value)
    assert result.value > 0


# Scale to Local Density Tests


@pytest.mark.parametrize("profile", dm_profiles)
def test_profiles_scale_to_local_density(profile):
    """Test scaling to local density with default local_density."""
    p = profile()
    p.scale_to_local_density(distance=DISTANCE_REF)
    actual = p(DISTANCE_REF)
    desired = p.LOCAL_DENSITY
    assert_quantity_allclose(actual, desired)


@pytest.mark.parametrize("profile", dm_profiles)
def test_profiles_scale_to_local_density_custom(profile):
    """Test scaling to local density with custom distance and local_density."""
    p = profile()
    custom_distance = 50 * u.kpc
    custom_density = 0.1 * u.GeV / u.cm**3
    p.scale_to_local_density(distance=custom_distance, local_density=custom_density)
    actual = p(custom_distance)
    assert_quantity_allclose(actual, custom_density)


@pytest.mark.parametrize("profile", dm_profiles)
def test_profiles_scale_to_local_density_dsph(profile):
    """Test scaling for a typical dSph use case."""
    p = profile()
    dsph_distance = 80 * u.kpc  # e.g. Sculptor dSph
    dsph_density = 0.5 * u.GeV / u.cm**3
    p.scale_to_local_density(distance=dsph_distance, local_density=dsph_density)
    actual = p(dsph_distance)
    assert_quantity_allclose(actual, dsph_density)


# Parameter Tests


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
    assert p.parameters["alpha"].value == pytest.approx(alpha_custom)
    assert p.parameters["beta"].value == pytest.approx(beta_custom)
    assert p.parameters["gamma"].value == pytest.approx(gamma_custom)
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


# Evaluation Tests


def test_zhao_profile_evaluate():
    """Test ZhaoProfile evaluation at different radii."""
    p = profiles.ZhaoProfile()

    r_s = p.parameters["r_s"].quantity
    result = p(r_s)

    assert result.unit.is_equivalent(u.GeV / u.cm**3)
    assert result.value > 0


def test_nfw_profile_evaluate():
    """Test NFWProfile evaluation."""
    p = profiles.NFWProfile()
    r_s = p.parameters["r_s"].quantity
    rho_s = p.parameters["rho_s"].quantity

    result = p(r_s)
    expected = rho_s / 4.0

    assert_quantity_allclose(result, expected)


def test_nfw_profile_evaluate_asymptotic():
    """Test NFWProfile asymptotic behavior."""
    p = profiles.NFWProfile()
    r_s = p.parameters["r_s"].quantity

    small_r = r_s * 0.001
    result_small = p(small_r)

    large_r = r_s * 1000
    result_large = p(large_r)

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

    result = p(r_s)
    expected = rho_s / 2.0

    assert_quantity_allclose(result, expected)


def test_burkert_profile_evaluate():
    """Test BurkertProfile evaluation."""
    p = profiles.BurkertProfile()
    r_s = p.parameters["r_s"].quantity
    rho_s = p.parameters["rho_s"].quantity

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


# Monotonic Behavior Tests


@pytest.mark.parametrize("profile", dm_profiles)
def test_profile_monotonic_decreasing(profile):
    """Test that density decreases monotonically with radius."""
    p = profile()

    radii = np.logspace(-2, 3, 50) * u.kpc
    densities = np.array([p(r).value for r in radii])

    assert np.all(np.diff(densities) < 0), "Density should decrease with radius"


# Integration Tests


def test_nfw_profile_integral():
    """Test integration of NFW profile."""
    p = profiles.NFWProfile()

    rmin = 0.01 * u.kpc
    rmax = 100 * u.kpc
    separation = 0.0
    ndecade = 100

    result_squared = p.integral(
        rmin, rmax, separation, ndecade, distance=DISTANCE_INTEGRAL, squared=True
    )
    assert result_squared.unit.is_equivalent(u.GeV**2 / u.cm**5)
    assert result_squared.value > 0

    result_unsquared = p.integral(
        rmin, rmax, separation, ndecade, distance=DISTANCE_INTEGRAL, squared=False
    )
    assert result_unsquared.unit.is_equivalent(u.GeV / u.cm**2)
    assert result_unsquared.value > 0

    assert result_squared.value > result_unsquared.value


def test_einasto_profile_integral():
    """Test integration of Einasto profile."""
    p = profiles.EinastoProfile()

    rmin = 0.01 * u.kpc
    rmax = 100 * u.kpc
    separation = 0.0
    ndecade = 100

    result = p.integral(
        rmin, rmax, separation, ndecade, distance=DISTANCE_INTEGRAL, squared=True
    )
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

    result = p.integral(
        rmin, rmax, separation, ndecade, distance=DISTANCE_INTEGRAL, squared=True
    )

    assert result.unit.is_equivalent(u.GeV**2 / u.cm**5)
    assert result.value > 0


# Call Method Tests


@pytest.mark.parametrize("profile", dm_profiles)
def test_profile_call_method(profile):
    """Test that __call__ method works correctly."""
    p = profile()
    r = 10 * u.kpc

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


# Special Cases and Edge Cases


def test_zhao_is_nfw_special_case():
    """Test that Zhao profile with (alpha=1, beta=3, gamma=1) is equivalent to NFW."""
    zhao = profiles.ZhaoProfile(alpha=1, beta=3, gamma=1)
    nfw = profiles.NFWProfile()

    test_radii = np.array([0.1, 1, 10, 100]) * u.kpc

    for r in test_radii:
        zhao_result = zhao(r)
        nfw_result = nfw(r)
        assert np.isclose(zhao_result.value, nfw_result.value, rtol=1e-10)


def test_profile_with_zero_radius_handling():
    """Test profile behavior near zero radius."""
    p = profiles.NFWProfile()

    r_small = 1e-6 * u.kpc
    result = p(r_small)

    assert result.value > 0
    assert np.isfinite(result.value)


@pytest.mark.parametrize("profile", dm_profiles)
def test_profile_parameter_modification(profile):
    """Test that modifying parameters affects the profile evaluation."""
    p = profile()
    r = 10 * u.kpc

    result_initial = p(r)

    original_rho_s = p.parameters["rho_s"].value
    p.parameters["rho_s"].value = original_rho_s * 2

    result_modified = p(r)

    assert np.isclose(result_modified.value / result_initial.value, 2.0, rtol=1e-10)


# Distance Parameter Tests


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

    separation_valid = 0.0
    result = p.integral(
        rmin, rmax, separation_valid, ndecade, distance=DISTANCE_INTEGRAL, squared=True
    )
    assert np.isfinite(result.value)
    assert result.value > 0

    separation_small = 1e-6
    result_small = p.integral(
        rmin, rmax, separation_small, ndecade, distance=DISTANCE_INTEGRAL, squared=True
    )
    assert np.isfinite(result_small.value)


def test_nfw_integral_with_custom_distance():
    """Test NFW integration with two different distances to target."""
    p = profiles.NFWProfile()

    rmin = 0.01 * u.kpc
    rmax = 100 * u.kpc
    separation = 0.5
    ndecade = 50

    distance_1 = DISTANCE_INTEGRAL
    result_1 = p.integral(
        rmin, rmax, separation, ndecade, distance=distance_1, squared=True
    )

    distance_2 = 10 * u.kpc
    result_2 = p.integral(
        rmin, rmax, separation, ndecade, distance=distance_2, squared=True
    )

    assert result_1.value != result_2.value


# HTML Representation Tests


@pytest.mark.parametrize("profile", dm_profiles)
def test_profile_html_repr(profile):
    """Test that profiles have working HTML representation."""
    p = profile()

    html_repr = p._repr_html_()
    assert isinstance(html_repr, str)
    assert len(html_repr) > 0


# Numerical Consistency Tests


def test_nfw_vs_evaluate_method():
    """Test that __call__ and evaluate methods give same results."""
    p = profiles.NFWProfile(r_s=20 * u.kpc, rho_s=0.5 * u.Unit("GeV / cm3"))

    r = 15 * u.kpc

    result_call = p(r)

    result_evaluate = profiles.NFWProfile.evaluate(
        r, p.parameters["r_s"].quantity, p.parameters["rho_s"].quantity
    )

    assert_quantity_allclose(result_call, result_evaluate)


# Dimensionality Tests


@pytest.mark.parametrize("profile", dm_profiles)
def test_profile_dimensional_consistency(profile):
    """Test that profile maintains dimensional consistency."""
    p = profile()

    r_kpc = 10 * u.kpc
    r_pc = r_kpc.to(u.pc)

    result_kpc = p(r_kpc)
    result_pc = p(r_pc)

    assert_quantity_allclose(result_kpc, result_pc, rtol=1e-10)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import html
from unittest.mock import patch

import astropy.units as u
import pytest
from numpy.testing import assert_allclose

from gammapy.astro.darkmatter import (
    DarkMatterAnnihilationSpectralModel,
    DarkMatterDecaySpectralModel,
    JFactory,
    LogNormalNuisancePrior,
    profiles,
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
        distance=profiles.DMProfile.DISTANCE_GC,
    )
    return jfactory.compute_jfactor()


@pytest.fixture(scope="session")
def jfact_decay(geom):
    jfactory = JFactory(
        geom=geom,
        profile=profiles.NFWProfile(),
        distance=profiles.DMProfile.DISTANCE_GC,
        annihilation=False,
    )
    return jfactory.compute_jfactor()


@requires_data()
def test_dmfluxmap_annihilation(jfact_annihilation):
    energy_min = 0.1 * u.TeV
    energy_max = 10 * u.TeV
    massDM = 1 * u.TeV
    channel = "W"

    diff_flux = DarkMatterAnnihilationSpectralModel(mass=massDM, channel=channel)
    int_flux = (
        jfact_annihilation
        * diff_flux.integral(energy_min=energy_min, energy_max=energy_max)
        / diff_flux.jfactor
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
    desired = 1.6796e-3 / u.cm**2 / u.s
    assert_quantity_allclose(actual, desired, rtol=1e-3)


def test_jfactory_repr_html_fallback(geom):
    jfactory = JFactory(geom=geom, profile=profiles.NFWProfile(), distance=8.5 * u.kpc)

    with patch.object(jfactory, "to_html", side_effect=AttributeError, create=True):
        repr_str = jfactory._repr_html_()

        # Check that it went through the except block returning the <pre> tags
        assert repr_str.startswith("<pre>")
        assert repr_str.endswith("</pre>")

        expected_string = html.escape(str(jfactory))
        assert expected_string in repr_str


def test_lognormal_prior_evaluate_at_center():
    """Prior must be zero at the observed value."""
    prior = LogNormalNuisancePrior(log10_obs=19.3, sigma_stat=0.1)
    test_val = Parameter(name="test_val", value=19.3)
    assert_allclose(prior(test_val), 0.0, atol=1e-10)


def test_lognormal_prior_evaluate_one_sigma():
    """Prior must be 1.0 exactly one sigma away."""
    prior = LogNormalNuisancePrior(log10_obs=19.3, sigma_stat=0.1)
    assert_allclose(prior(Parameter(name="test_val", value=19.4)), 1.0, rtol=1e-5)
    assert_allclose(prior(Parameter(name="test_val", value=19.2)), 1.0, rtol=1e-5)


def test_lognormal_prior_sigma_in_quadrature():
    """sigma_total must be computed as sqrt(stat² + syst²)."""
    prior = LogNormalNuisancePrior(log10_obs=19.3, sigma_stat=0.3, sigma_syst=0.4)
    assert_allclose(prior.sigma_total.value, 0.5, rtol=1e-6)


def test_lognormal_prior_only_syst():
    prior = LogNormalNuisancePrior(log10_obs=19.3, sigma_syst=0.2)
    assert_allclose(prior.sigma_total.value, 0.2, rtol=1e-6)
    assert_allclose(prior(Parameter(name="test_val", value=19.5)), 1.0, rtol=1e-5)


def test_lognormal_prior_invalid_sigma_stat():
    with pytest.raises((TypeError, ValueError)):
        LogNormalNuisancePrior(log10_obs=19.3, sigma_stat=-0.1)

    with pytest.raises((TypeError, ValueError)):
        LogNormalNuisancePrior(log10_obs=19.3, sigma_stat="bad")


def test_lognormal_prior_invalid_sigma_syst():
    with pytest.raises((TypeError, ValueError)):
        LogNormalNuisancePrior(log10_obs=19.3, sigma_syst="bad")

    with pytest.raises((TypeError, ValueError)):
        LogNormalNuisancePrior(log10_obs=19.3, sigma_syst=-0.1)

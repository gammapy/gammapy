# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import astropy.units as u
from gammapy.astro.darkmatter import (
    DarkMatterAnnihilationSpectralModel,
    DarkMatterDecaySpectralModel,
    JFactory,
    profiles,
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
        jfact_decay * diff_flux.integral(energy_min=energy_min, energy_max=energy_max)
    ).to("cm-2 s-1")
    actual = int_flux[5, 5]
    desired = 7.01927e-3 / u.cm**2 / u.s
    assert_quantity_allclose(actual, desired, rtol=1e-3)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import astropy.units as u
from gammapy.astro.darkmatter import (
    DarkMatterAnnihilationSpectralModel,
    JFactory,
    profiles,
)
from gammapy.maps import WcsGeom
from gammapy.utils.testing import assert_quantity_allclose, requires_data


@pytest.fixture(scope="session")
def geom():
    return WcsGeom.create(binsz=0.5, npix=10)


@pytest.fixture(scope="session")
def jfact(geom):
    jfactory = JFactory(geom=geom, profile=profiles.NFWProfile(), distance=8 * u.kpc)
    return jfactory.compute_jfactor()


@requires_data()
def test_dmfluxmap(jfact):
    energy_min = 0.1 * u.TeV
    energy_max = 10 * u.TeV
    massDM = 1 * u.TeV
    channel = "W"

    diff_flux = DarkMatterAnnihilationSpectralModel(mass=massDM, channel=channel)
    int_flux = (
        jfact * diff_flux.integral(energy_min=energy_min, energy_max=energy_max)
    ).to("cm-2 s-1")
    actual = int_flux[5, 5]
    desired = 1.9483e-12 / u.cm**2 / u.s
    assert_quantity_allclose(actual, desired, rtol=1e-3)

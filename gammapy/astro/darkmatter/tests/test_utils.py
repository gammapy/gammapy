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


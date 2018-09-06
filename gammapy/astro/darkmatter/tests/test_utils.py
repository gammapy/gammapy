# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import astropy.units as u
from ....utils.testing import (
    assert_quantity_allclose,
    requires_data,
    requires_dependency,
)
from ....maps import WcsGeom
from .. import JFactory, profiles, PrimaryFlux, compute_dm_flux


@pytest.fixture(scope="session")
def geom():
    return WcsGeom.create(binsz=0.5, npix=10)


@pytest.fixture(scope="session")
def jfact(geom):
    jfactory = JFactory(geom=geom, profile=profiles.NFWProfile(), distance=8 * u.kpc)
    return jfactory.compute_jfactor()


@pytest.fixture(scope="session")
def prim_flux():
    return PrimaryFlux(mDM=1 * u.TeV, channel="W")


@requires_dependency("scipy")
@requires_data("gammapy-extra")
def test_dmfluxmapmaker(jfact, prim_flux):
    x_section = 1e-26 * u.Unit("cm3 s-1")
    energy_range = [0.1, 1] * u.TeV
    flux = compute_dm_flux(
        jfact=jfact, prim_flux=prim_flux, x_section=x_section, energy_range=energy_range
    )

    actual = flux[5, 5]
    desired = 6.503327e-13 / u.cm ** 2 / u.s
    assert_quantity_allclose(actual, desired, rtol=1e-5)

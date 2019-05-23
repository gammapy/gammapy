# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import astropy.units as u
from ....utils.testing import assert_quantity_allclose, requires_data
from ....maps import WcsGeom
from .. import JFactory, profiles, DMAnnihilation


@pytest.fixture(scope="session")
def geom():
    return WcsGeom.create(binsz=0.5, npix=10)


@pytest.fixture(scope="session")
def jfact(geom):
    jfactory = JFactory(geom=geom, profile=profiles.NFWProfile(), distance=8 * u.kpc)
    return jfactory.compute_jfactor()


@requires_data()
def test_dmfluxmap(jfact):

    emin = 0.1 * u.TeV
    emax = 10 * u.TeV
    massDM = 1 * u.TeV
    channel = "W"

    diff_flux = DMAnnihilation(mass=massDM, channel=channel)
    int_flux = (jfact * diff_flux.integral(emin=emin, emax=emax)).to("cm-2 s-1")
    actual = int_flux[5, 5]
    desired = 1.94839226e-12 / u.cm ** 2 / u.s
    assert_quantity_allclose(actual, desired, rtol=1e-5)

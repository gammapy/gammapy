# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.table import Table
import astropy.units as u
from ..models import PowerLaw
from ..flux_point import FluxPoints


@pytest.fixture(scope="session")
def model():
    return PowerLaw()


@pytest.fixture(scope="session")
def flux_points_dnde(model):
    e_ref = [np.sqrt(10), np.sqrt(10 * 100)] * u.TeV
    table = Table()
    table.meta["SED_TYPE"] = "dnde"
    table["e_ref"] = e_ref
    table["dnde"] = model(e_ref)
    return FluxPoints(table)


@pytest.fixture(scope="session")
def flux_points_e2dnde(model):
    e_ref = [np.sqrt(10), np.sqrt(10 * 100)] * u.TeV
    table = Table()
    table.meta["SED_TYPE"] = "e2dnde"
    table["e_ref"] = e_ref
    table["e2dnde"] = (model(e_ref) * e_ref ** 2).to("erg cm-2 s-1")
    return FluxPoints(table)


@pytest.fixture(scope="session")
def flux_points_flux(model):
    e_min = [1, 10] * u.TeV
    e_max = [10, 100] * u.TeV

    table = Table()
    table.meta["SED_TYPE"] = "flux"
    table["e_min"] = e_min
    table["e_max"] = e_max
    table["flux"] = model.integral(e_min, e_max)
    return FluxPoints(table)


def test_dnde_to_e2dnde(flux_points_dnde, flux_points_e2dnde):
    actual = flux_points_dnde.to_sed_type("e2dnde").table
    desired = flux_points_e2dnde.table
    assert_allclose(actual["e2dnde"], desired["e2dnde"])


def test_e2dnde_to_dnde(flux_points_e2dnde, flux_points_dnde):
    actual = flux_points_e2dnde.to_sed_type("dnde").table
    desired = flux_points_dnde.table
    assert_allclose(actual["dnde"], desired["dnde"])


def test_flux_to_dnde(flux_points_flux, flux_points_dnde):
    actual = flux_points_flux.to_sed_type("dnde", method="log_center").table
    desired = flux_points_dnde.table
    assert_allclose(actual["e_ref"], desired["e_ref"])
    assert_allclose(actual["dnde"], desired["dnde"])

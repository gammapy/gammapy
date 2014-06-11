# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.tests.helper import pytest
from .. import simulate

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.fixture
def example_table():
    from ..spatial import YK04
    from ..velocity import H05
    nsources = 42
    max_age = 1e6
    return simulate.make_cat_gal(nsources=nsources, rad_dis=YK04, vel_dis=H05, max_age=max_age)


def has_columns(table, names):
    """Helper function to check if a table has a list of columns."""
    return set(names).issubset(table.colnames)


def test_make_cat_cube():
    nsources = 100
    table = simulate.make_cat_cube(nsources=nsources)
    assert len(table) == nsources


def test_make_cat_gal():
    from ..spatial import YK04
    from ..velocity import H05
    nsources = 42
    max_age = 1e6

    table = simulate.make_cat_gal(nsources=nsources, rad_dis=YK04, vel_dis=H05, max_age=max_age)
    assert len(table) == nsources


def test_add_par_snr(example_table):
    table = simulate.add_par_snr(example_table)
    assert len(table) == len(example_table)
    assert has_columns(table, ['E_SN'])


def test_add_par_psr(example_table):
    table = simulate.add_par_psr(example_table)
    assert len(table) == len(example_table)
    assert has_columns(table, ['P0'])


@pytest.mark.skipif('not HAS_SCIPY')
def test_add_par_pwn(example_table):
    # To compute PWN parameters we need PSR and SNR parameters first
    table = simulate.add_par_snr(example_table)
    table = simulate.add_par_psr(table)
    table = simulate.add_par_pwn(table)
    assert len(table) == len(example_table)
    assert has_columns(table, ['r_out_PWN'])


@pytest.mark.skipif('not HAS_SCIPY')
def test_add_par_obs(example_table):
    table = simulate.add_par_snr(example_table)
    table = simulate.add_par_psr(table)
    table = simulate.add_par_pwn(table)
    table = simulate.add_observed_parameters(table)
    table = simulate.add_par_obs(table)
    assert len(table) == len(example_table)
    assert has_columns(table, ['ext_in_SNR'])


def test_add_observed_parameters(example_table):
    table = simulate.add_cylindrical_coordinates(example_table)
    assert len(table) == len(example_table)
    assert has_columns(table, ['r', 'phi'])

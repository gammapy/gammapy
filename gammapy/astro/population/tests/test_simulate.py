# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.tests.helper import pytest
from ...population import (make_base_catalog_galactic,
                           make_cat_cube,
                           add_snr_parameters,
                           add_pulsar_parameters,
                           add_pwn_parameters,
                           add_observed_parameters,
                           add_observed_source_parameters,
                           )

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.fixture
def example_table():
    from ..spatial import YusifovKucuk2004
    from ..velocity import FaucherKaspi2006VelocityMaxwellian
    rad_dis = YusifovKucuk2004
    vel_dis = FaucherKaspi2006VelocityMaxwellian
    n_sources = 42
    max_age = 1e6
    return make_base_catalog_galactic(n_sources=n_sources, rad_dis=rad_dis,
                                      vel_dis=vel_dis, max_age=max_age)


def has_columns(table, names):
    """Helper function to check if a table has a list of columns."""
    return set(names).issubset(table.colnames)


def test_make_cat_cube():
    n_sources = 100
    table = make_cat_cube(n_sources=n_sources)
    assert len(table) == n_sources


def test_make_cat_gal():
    from ..spatial import YusifovKucuk2004
    from ..velocity import FaucherKaspi2006VelocityMaxwellian
    rad_dis = YusifovKucuk2004
    vel_dis = FaucherKaspi2006VelocityMaxwellian
    n_sources = 42
    max_age = 1e6

    table = make_base_catalog_galactic(n_sources=n_sources, rad_dis=rad_dis,
                                       vel_dis=vel_dis, max_age=max_age)
    assert len(table) == n_sources


def test_add_snr_parameters(example_table):
    table = add_snr_parameters(example_table)
    assert len(table) == len(example_table)
    assert has_columns(table, ['E_SN'])


def test_add_pulsar_parameters(example_table):
    table = add_pulsar_parameters(example_table)
    assert len(table) == len(example_table)
    assert has_columns(table, ['P0'])


@pytest.mark.skipif('not HAS_SCIPY')
def test_add_pwn_parameters(example_table):
    # To compute PWN parameters we need PSR and SNR parameters first
    table = add_snr_parameters(example_table)
    table = add_pulsar_parameters(table)
    table = add_pwn_parameters(table)
    assert len(table) == len(example_table)
    assert has_columns(table, ['r_out_PWN'])


@pytest.mark.skipif('not HAS_SCIPY')
def test_add_par_obs(example_table):
    table = add_snr_parameters(example_table)
    table = add_pulsar_parameters(table)
    table = add_pwn_parameters(table)
    table = add_observed_parameters(table)
    table = add_observed_source_parameters(table)
    assert len(table) == len(example_table)
    assert has_columns(table, ['ext_in_SNR'])

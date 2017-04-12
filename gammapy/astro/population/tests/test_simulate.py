# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ....utils.testing import requires_dependency
from ...population import (
    make_base_catalog_galactic,
    make_catalog_random_positions_cube,
    make_catalog_random_positions_sphere,
    add_snr_parameters,
    add_pulsar_parameters,
    add_pwn_parameters,
    add_observed_parameters,
    add_observed_source_parameters,
)


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


def test_make_catalog_random_positions_cube():
    size = 100
    table = make_catalog_random_positions_cube(size=size)
    assert len(table) == size


def test_make_catalog_random_positions_sphere():
    size = 100
    table = make_catalog_random_positions_sphere(size=size,
                                                 center='Milky Way')
    assert len(table) == size


def test_make_base_catalog_galactic():
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


@requires_dependency('scipy')
def test_add_pwn_parameters(example_table):
    # To compute PWN parameters we need PSR and SNR parameters first
    table = add_snr_parameters(example_table)
    table = add_pulsar_parameters(table)
    table = add_pwn_parameters(table)
    assert len(table) == len(example_table)
    assert has_columns(table, ['r_out_PWN'])


@requires_dependency('scipy')
def test_add_par_obs(example_table):
    table = add_snr_parameters(example_table)
    table = add_pulsar_parameters(table)
    table = add_pwn_parameters(table)
    table = add_observed_parameters(table)
    table = add_observed_source_parameters(table)
    assert len(table) == len(example_table)
    assert has_columns(table, ['ext_in_SNR'])


def test_make_base_catalog_galactic():
    """Test that make_base_catalog_galactic uses random_state correctly.
    
    Calling with a given seed should always give the same output.

    Regression test for https://github.com/gammapy/gammapy/issues/959
    """
    table = make_base_catalog_galactic(n_sources=1, random_state=0)
    d = table[0]
    # print(list(zip(d.colnames, d.as_void())))

    assert_allclose(d['x_birth'], -7.7244510367459904)
    assert_allclose(d['y_birth'], -0.3878120924926074)
    assert_allclose(d['z_birth'], 0.028117277025426112)
    assert_allclose(d['x'], -7.7244510367459904)
    assert_allclose(d['y'], -0.3878120924926074)
    assert_allclose(d['z'], 0.028117277025426112)
    assert_allclose(d['vx'], -341.28926571465314)
    assert_allclose(d['vy'], 48.211461842444507)
    assert_allclose(d['vz'], 298.79950689388124)

    assert_allclose(d['age'], -0.056712977317443181)
    assert_allclose(d['n_ISM'], 1.0)
    assert d['spiralarm'] == 'Crux Scutum'
    assert d['morph_type'] == 'shell2d'
    assert_allclose(d['v_abs'], 456.16209099952528)

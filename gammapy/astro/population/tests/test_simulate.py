# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
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
    """Test that make_base_catalog_galactic uses random_state correctly.

    Calling with a given seed should always give the same output.

    Regression test for https://github.com/gammapy/gammapy/issues/959
    """
    table = make_base_catalog_galactic(n_sources=10, random_state=0)
    assert len(table) == 10
    assert table.colnames == [
        'x_birth', 'y_birth', 'z_birth', 'x', 'y', 'z',
        'vx', 'vy', 'vz', 'age', 'n_ISM', 'spiralarm', 'morph_type', 'v_abs'
    ]

    d = table[0]
    # print(list(zip(d.colnames, d.as_void())))

    assert_allclose(d['x_birth'], 0.58513884523635529)
    assert_allclose(d['y_birth'], -11.682838075998815)
    assert_allclose(d['z_birth'], 0.15710260060554912)
    assert_allclose(d['x'], 0.58513884523635529)
    assert_allclose(d['y'], -11.682838075998815)
    assert_allclose(d['z'], 0.15710260060554912)
    assert_allclose(d['vx'], -4.1266001441394655)
    assert_allclose(d['vy'], 42.543357869627776)
    assert_allclose(d['vz'], 345.43206179709432)

    assert_allclose(d['age'], -0.54881350392732475)  # TODO: why negative?
    assert_allclose(d['n_ISM'], 1.0)
    assert d['spiralarm'] == 'Crux Scutum'
    assert d['morph_type'] == 'shell2d'
    assert_allclose(d['v_abs'], 348.06648135803658)


def test_add_snr_parameters():
    table = make_base_catalog_galactic(n_sources=10, random_state=0)
    table = add_snr_parameters(table)
    assert len(table) == 10
    assert set(['E_SN']) < set(table.colnames)

    d = table[0]
    # print(list(zip(d.colnames, d.as_void())))
    assert_allclose(d['E_SN'], 9.9999999999999999e+50)
    assert_allclose(d['r_out'], -0.0054881350392732477)  # TODO: why negative?
    assert_allclose(d['r_in'], -0.0049865194966836725)  # TODO: why negative?
    assert_allclose(d['L_SNR'], 0)  # TODO: why zero?


def test_add_pulsar_parameters():
    table = make_base_catalog_galactic(n_sources=10, random_state=0)
    table = add_pulsar_parameters(table, random_state=0)
    assert len(table) == 10
    assert set(['P0']) < set(table.colnames)

    d = table[0]
    # print(list(zip(d.colnames, d.as_void())))
    assert_allclose(d['P0'], 0.32225433462770614)
    assert_allclose(d['P1'], 5.494766020259218e-15)
    assert_allclose(d['P0_birth'], 0.32225471528776467)
    assert_allclose(d['P1_birth'], 5.494759529623533e-15)
    assert_allclose(d['CharAge'], 2.8055241333205992e-23)
    assert_allclose(d['Tau0'], 929215.88847248629)
    assert_allclose(d['L_PSR'], 6.4820156070136442e+33)
    assert_allclose(d['L0_PSR'], 6.4820232638367655e+33)
    assert_allclose(d['logB'], 12.129223964138484)


@requires_dependency('scipy')
def test_add_pwn_parameters():
    table = make_base_catalog_galactic(n_sources=10, random_state=0)
    # To compute PWN parameters we need PSR and SNR parameters first
    table = add_snr_parameters(table)
    table = add_pulsar_parameters(table, random_state=0)
    table = add_pwn_parameters(table)
    assert len(table) == 10
    assert set(['r_out_PWN']) < set(table.colnames)

    d = table[0]
    # print(list(zip(d.colnames, d.as_void())))
    assert_allclose(d['r_out_PWN'], np.nan)  # TODO: why NaN???
    assert_allclose(d['L_PWN'], -1.122637636555517e+40)  # TODO: why negative?


@requires_dependency('scipy')
def test_chain_all():
    table = make_base_catalog_galactic(n_sources=10, random_state=0)
    table = add_snr_parameters(table)
    table = add_pulsar_parameters(table, random_state=0)
    table = add_pwn_parameters(table)
    table = add_observed_parameters(table)
    table = add_observed_source_parameters(table)
    assert len(table) == 10
    assert set(['ext_in_SNR']) < set(table.colnames)

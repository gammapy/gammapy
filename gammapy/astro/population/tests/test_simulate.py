# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.table import Table
import astropy.units as u
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
    table = make_catalog_random_positions_sphere(size=size, center="Milky Way")
    assert len(table) == size


def test_make_base_catalog_galactic():
    """Test that make_base_catalog_galactic uses random_state correctly.

    Calling with a given seed should always give the same output.

    Regression test for https://github.com/gammapy/gammapy/issues/959
    """
    table = make_base_catalog_galactic(n_sources=10, random_state=0)
    assert len(table) == 10
    assert table.colnames == [
        "age",
        "n_ISM",
        "spiralarm",
        "x_birth",
        "y_birth",
        "z_birth",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "v_abs",
    ]

    d = table[0]

    assert_allclose(d["age"], 548813.50392732478)
    assert_allclose(d["n_ISM"], 1.0)
    assert d["spiralarm"] == "Crux Scutum"

    assert_allclose(d["x_birth"], 0.58513884292018437)
    assert_allclose(d["y_birth"], -11.682838052120154)
    assert_allclose(d["z_birth"], 0.15710279448905115)
    assert_allclose(d["x"], 0.5828226720259867)
    assert_allclose(d["y"], -11.658959390801584)
    assert_allclose(d["z"], 0.35098629652725671)
    assert_allclose(d["vx"], -4.1266001441394655)
    assert_allclose(d["vy"], 42.543357869627776)
    assert_allclose(d["vz"], 345.43206179709432)
    assert_allclose(d["v_abs"], 348.06648135803658)


def test_add_observed_parameters():
    table = make_base_catalog_galactic(n_sources=10, random_state=0)
    table = add_observed_parameters(table)

    assert len(table) == 10
    assert set(table.colnames).issuperset(
        ["distance", "GLON", "GLAT", "VGLON", "VGLAT", "RA", "DEC"]
    )

    d = table[0]

    assert_allclose(d["distance"], 3231.392591455106)
    assert_allclose(d["GLON"], 169.54657778189639)
    assert_allclose(d["GLAT"], 6.2356357665816162)
    assert_allclose(d["VGLON"], 0.066778795313076678)
    assert_allclose(d["VGLAT"], 5.6115948931932174)
    assert_allclose(d["RA"], 86.308826288823127)
    assert_allclose(d["DEC"], 41.090120056648828)


def test_add_snr_parameters():
    table = Table()
    table["age"] = [100, 1000] * u.yr
    table["n_ISM"] = u.Quantity(1, "cm-3")

    table = add_snr_parameters(table)

    assert len(table) == 2
    assert table.colnames == ["age", "n_ISM", "E_SN", "r_out", "r_in", "L_SNR"]

    assert_allclose(table["E_SN"], 1e51)
    assert_allclose(table["r_out"], [1, 3.80730787743])
    assert_allclose(table["r_in"], [0.9086, 3.45931993743])
    assert_allclose(table["L_SNR"], [0, 1.0768e+33])


def test_add_pulsar_parameters():
    table = Table()
    table["age"] = [100, 1000] * u.yr

    table = add_pulsar_parameters(table, random_state=0)

    assert len(table) == 2
    assert table.colnames == [
        "age",
        "P0",
        "P1",
        "P0_birth",
        "P1_birth",
        "CharAge",
        "Tau0",
        "L_PSR",
        "L0_PSR",
        "logB",
    ]

    assert_allclose(table["P0"], [0.322829453422, 0.51352778881])
    assert_allclose(table["P1"], [4.54295751161e-14, 6.98423128444e-13])
    assert_allclose(table["P0_birth"], [0.322254715288, 0.388110930459])
    assert_allclose(table["P1_birth"], [4.55105983192e-14, 9.24116423053e-13])
    assert_allclose(table["CharAge"], [2.32368825638e-22, 5.6826197937e-21])
    assert_allclose(table["Tau0"], [112189.64476, 6654.19039158])
    assert_allclose(table["L_PSR"], [5.37834069771e+34, 8.25708734631e+35])
    assert_allclose(table["L0_PSR"], [5.36876555682e+34, 6.24049160082e+35])
    assert_allclose(table["logB"], [12.5883058913, 13.2824912596])


@requires_dependency("scipy")
def test_add_pwn_parameters():
    table = make_base_catalog_galactic(n_sources=10, random_state=0)
    # To compute PWN parameters we need PSR and SNR parameters first
    table = add_snr_parameters(table)
    table = add_pulsar_parameters(table, random_state=0)
    table = add_pwn_parameters(table)
    assert len(table) == 10

    d = table[0]
    assert_allclose(d["r_out_PWN"], 0.5892196771927385, atol=1e-3)
    assert_allclose(d["L_PWN"], 7.057857699785925e+45)


@requires_dependency("scipy")
def test_chain_all():
    """
    Test that running the simulation functions in chain works
    """
    table = make_base_catalog_galactic(n_sources=10, random_state=0)
    table = add_snr_parameters(table)
    table = add_pulsar_parameters(table, random_state=0)
    table = add_pwn_parameters(table)
    table = add_observed_parameters(table)
    table = add_observed_source_parameters(table)

    # Note: the individual functions are tested above.
    # Here we just run them in a chain and do very basic asserts
    # on the output so that we make sure we notice changes.
    assert len(table) == 10
    assert len(table.colnames) == 43
    d = table[0]
    assert_allclose(d["r_out_PWN"], 0.5892196771927385, atol=1e-3)
    assert_allclose(d["RA"], 86.308826288823127)

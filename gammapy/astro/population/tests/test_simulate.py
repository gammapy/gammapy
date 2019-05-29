# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose, assert_equal
from astropy.table import Table
import astropy.units as u
from ...population import (
    make_base_catalog_galactic,
    make_catalog_random_positions_cube,
    make_catalog_random_positions_sphere,
    add_snr_parameters,
    add_pulsar_parameters,
    add_pwn_parameters,
    add_observed_parameters,
)


def test_make_catalog_random_positions_cube():
    table = make_catalog_random_positions_cube(random_state=0)
    d = table[0]

    assert len(table) == 100
    assert len(table.colnames) == 3

    assert table["x"].unit == "pc"
    assert_allclose(d["x"], 0.0976270078546495)
    assert table["y"].unit == "pc"
    assert_allclose(d["y"], 0.3556330735924602)
    assert table["z"].unit == "pc"
    assert_allclose(d["z"], -0.37640823601179485)

    table = make_catalog_random_positions_cube(dimension=2, random_state=0)
    assert_equal(table["z"], 0)

    table = make_catalog_random_positions_cube(dimension=1, random_state=0)
    assert_equal(table["y"], 0)
    assert_equal(table["z"], 0)


def test_make_catalog_random_positions_sphere():
    table = make_catalog_random_positions_sphere(random_state=0)
    d = table[0]

    assert len(table) == 100
    assert len(table.colnames) == 3

    assert table["lon"].unit == "rad"
    assert_allclose(d["lon"], 3.4482969442579128)
    assert table["lat"].unit == "rad"
    assert_allclose(d["lat"], 0.36359133530192267)
    assert table["distance"].unit == "pc"
    assert_allclose(d["distance"], 0.6780943487897606)


def test_make_base_catalog_galactic():
    table = make_base_catalog_galactic(n_sources=10, random_state=0)
    d = table[0]

    assert len(table) == 10
    assert len(table.colnames) == 13

    assert table["age"].unit == "yr"
    assert_allclose(d["age"], 548813.50392732478)
    assert table["n_ISM"].unit == "cm-3"
    assert_allclose(d["n_ISM"], 1.0)
    assert table["spiralarm"].unit is None
    assert d["spiralarm"] == "Perseus"
    assert table["x_birth"].unit == "kpc"
    assert_allclose(d["x_birth"], -2.57846191600981)
    assert table["y_birth"].unit == "kpc"
    assert_allclose(d["y_birth"], 10.010139593948578)
    assert table["z_birth"].unit == "kpc"
    assert_allclose(d["z_birth"], -0.056572220998886355)
    assert table["x"].unit == "kpc"
    assert_allclose(d["x"], -2.580778086904008)
    assert table["y"].unit == "kpc"
    assert_allclose(d["y"], 10.034018255267148)
    assert table["z"].unit == "kpc"
    assert_allclose(d["z"], 0.13731128103931922)
    assert table["vx"].unit == "km/s"
    assert_allclose(d["vx"], -4.1266001441394655)
    assert table["vy"].unit == "km/s"
    assert_allclose(d["vy"], 42.543357869627776)
    assert table["vz"].unit == "km/s"
    assert_allclose(d["vz"], 345.4320617970943)
    assert table["v_abs"].unit == "km/s"
    assert_allclose(d["v_abs"], 348.06648135803658)


def test_add_snr_parameters():
    table = Table()
    table["age"] = [100, 1000] * u.yr
    table["n_ISM"] = u.Quantity(1, "cm-3")

    table = add_snr_parameters(table)

    assert len(table) == 2
    assert table.colnames == ["age", "n_ISM", "E_SN", "r_out", "r_in", "L_SNR"]

    assert table["E_SN"].unit == "erg"
    assert_allclose(table["E_SN"], 1e51)
    assert table["r_out"].unit == "pc"
    assert_allclose(table["r_out"], [1, 3.80730787743])
    assert table["r_in"].unit == "pc"
    assert_allclose(table["r_in"], [0.9086, 3.45931993743])
    assert table["L_SNR"].unit == "1 / s"
    assert_allclose(table["L_SNR"], [0, 1.0768e33])


def test_add_pulsar_parameters():
    table = Table()
    table["age"] = [100, 1000] * u.yr

    table = add_pulsar_parameters(table, random_state=0)

    assert len(table) == 2
    assert len(table.colnames) == 10

    assert table["age"].unit == "yr"
    assert_allclose(table["age"], [100, 1000])
    assert table["P0"].unit == "s"
    assert_allclose(table["P0"], [0.322829453422, 0.51352778881])
    assert table["P1"].unit == ""
    assert_allclose(table["P1"], [4.54295751161e-14, 6.98423128444e-13])
    assert table["P0_birth"].unit == "s"
    assert_allclose(table["P0_birth"], [0.322254715288, 0.388110930459])
    assert table["P1_birth"].unit == ""
    assert_allclose(table["P1_birth"], [4.55105983192e-14, 9.24116423053e-13])
    assert table["CharAge"].unit == "yr"
    assert_allclose(table["CharAge"], [2.32368825638e-22, 5.6826197937e-21])
    assert table["Tau0"].unit == "yr"
    assert_allclose(table["Tau0"], [112189.64476, 6654.19039158])
    assert table["L_PSR"].unit == "erg / s"
    assert_allclose(table["L_PSR"], [5.37834069771e34, 8.25708734631e35])
    assert table["L0_PSR"].unit == "erg / s"
    assert_allclose(table["L0_PSR"], [5.36876555682e34, 6.24049160082e35])
    assert (
        table["logB"].unit == "G"
    )  # TODO: logB should be dimensionless. Fix? Change to "B"?
    assert_allclose(table["logB"], [12.5883058913, 13.2824912596])


def test_add_pwn_parameters():
    table = make_base_catalog_galactic(n_sources=10, random_state=0)
    # To compute PWN parameters we need PSR and SNR parameters first
    table = add_snr_parameters(table)
    table = add_pulsar_parameters(table, random_state=0)
    table = add_pwn_parameters(table)
    d = table[0]

    assert len(table) == 10
    assert len(table.colnames) == 27

    assert table["r_out_PWN"].unit == "pc"
    assert_allclose(d["r_out_PWN"], 0.5892196771927385, atol=1e-3)


def test_add_observed_parameters():
    table = make_base_catalog_galactic(n_sources=10, random_state=0)
    table = add_observed_parameters(table)
    d = table[0]

    assert len(table) == 10
    assert len(table.colnames) == 20

    assert table["distance"].unit == "pc"
    assert_allclose(d["distance"], 18713.340231191236)
    assert table["GLON"].unit == "deg"
    assert_allclose(d["GLON"], -7.9272056829002615)
    assert table["GLAT"].unit == "deg"
    assert_allclose(d["GLAT"], 0.42041812871409573)
    assert table["VGLON"].unit == "deg / Myr"
    assert_allclose(d["VGLON"], -0.005574473038475241)
    assert table["VGLAT"].unit == "deg / Myr"
    assert_allclose(d["VGLAT"], 1.0736832036530914)
    assert table["RA"].unit == "deg"
    assert_allclose(d["RA"], 260.9075160323843)
    assert table["DEC"].unit == "deg"
    assert_allclose(d["DEC"], -35.371104990114816)


def test_chain_all():
    # Test that running the simulation functions in chain works
    table = make_base_catalog_galactic(n_sources=10, random_state=0)
    table = add_snr_parameters(table)
    table = add_pulsar_parameters(table, random_state=0)
    table = add_pwn_parameters(table)
    table = add_observed_parameters(table)
    d = table[0]

    # Note: the individual functions are tested above.
    # Here we just run them in a chain and do very basic asserts
    # on the output so that we make sure we notice changes.
    assert len(table) == 10
    assert len(table.colnames) == 34

    assert table["r_out_PWN"].unit == "pc"
    assert_allclose(d["r_out_PWN"], 0.5891300402092443)
    assert table["RA"].unit == "deg"
    assert_allclose(d["RA"], 260.9075160323843)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
from astropy.table import Table
from gammapy.astro.population import (
    add_observed_parameters,
    add_pulsar_parameters,
    add_pwn_parameters,
    add_snr_parameters,
    make_base_catalog_galactic,
    make_catalog_random_positions_cube,
    make_catalog_random_positions_sphere,
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
    assert d["spiralarm"] == "Crux Scutum"
    assert table["x_birth"].unit == "kpc"
    assert_allclose(d["x_birth"], -5.856461, atol=1e-5)
    assert table["y_birth"].unit == "kpc"
    assert_allclose(d["y_birth"], 3.017292, atol=1e-5)
    assert table["z_birth"].unit == "kpc"
    assert_allclose(d["z_birth"], 0.049088, atol=1e-5)
    assert table["x"].unit == "kpc"
    assert_allclose(d["x"], -5.941061, atol=1e-5)
    assert table["y"].unit == "kpc"
    assert_allclose(d["y"], 3.081642, atol=1e-5)
    assert table["z"].unit == "kpc"
    assert_allclose(d["z"], 0.023161, atol=1e-5)
    assert table["vx"].unit == "km/s"
    assert_allclose(d["vx"], -150.727104, atol=1e-5)
    assert table["vy"].unit == "km/s"
    assert_allclose(d["vy"], 114.648494, atol=1e-5)
    assert table["vz"].unit == "km/s"
    assert_allclose(d["vz"], -46.193814, atol=1e-5)
    assert table["v_abs"].unit == "km/s"
    assert_allclose(d["v_abs"], 194.927693, atol=1e-5)


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
    assert_allclose(table["P0"], [0.214478, 0.246349], atol=1e-5)
    assert table["P1"].unit == ""
    assert_allclose(table["P1"], [6.310423e-13, 4.198294e-16], atol=1e-5)
    assert table["P0_birth"].unit == "s"
    assert_allclose(table["P0_birth"], [0.212418, 0.246336], atol=1e-5)
    assert table["P1_birth"].unit == ""
    assert_allclose(table["P1_birth"], [6.558773e-13, 4.199198e-16], atol=1e-5)
    assert table["CharAge"].unit == "yr"
    assert_allclose(table["CharAge"], [2.207394e-21, 1.638930e-24], atol=1e-5)
    assert table["Tau0"].unit == "yr"
    assert_allclose(table["Tau0"], [5.131385e03, 9.294538e06], atol=1e-5)
    assert table["L_PSR"].unit == "erg / s"
    assert_allclose(table["L_PSR"], [2.599229e36, 1.108788e33], rtol=1e-5)
    assert table["L0_PSR"].unit == "erg / s"
    assert_allclose(table["L0_PSR"], [2.701524e36, 1.109026e33], rtol=1e-5)
    assert table["B_PSR"].unit == "G"
    assert_allclose(table["B_PSR"], [1.194420e13, 3.254597e11], rtol=1e-5)


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
    assert_allclose(d["r_out_PWN"], 1.378224, atol=1e-4)


def test_add_observed_parameters():
    table = make_base_catalog_galactic(n_sources=10, random_state=0)
    table = add_observed_parameters(table)
    d = table[0]

    assert len(table) == 10
    assert len(table.colnames) == 20

    assert table["distance"].unit == "pc"
    assert_allclose(d["distance"], 13016.572756, atol=1e-5)
    assert table["GLON"].unit == "deg"
    assert_allclose(d["GLON"], -27.156565, atol=1e-5)
    assert table["GLAT"].unit == "deg"
    assert_allclose(d["GLAT"], 0.101948, atol=1e-5)
    assert table["VGLON"].unit == "deg / Myr"
    assert_allclose(d["VGLON"], 0.368166, atol=1e-5)
    assert table["VGLAT"].unit == "deg / Myr"
    assert_allclose(d["VGLAT"], -0.209514, atol=1e-5)
    assert table["RA"].unit == "deg"
    assert_allclose(d["RA"], 244.347149, atol=1e-5)
    assert table["DEC"].unit == "deg"
    assert_allclose(d["DEC"], -50.410142, atol=1e-5)


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
    assert_allclose(d["r_out_PWN"], 1.378224, atol=1e-4)
    assert table["RA"].unit == "deg"
    assert_allclose(d["RA"], 244.347149, atol=1e-5)

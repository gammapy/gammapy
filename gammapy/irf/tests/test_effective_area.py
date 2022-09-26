# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
from gammapy.irf import EffectiveAreaTable2D
from gammapy.maps import MapAxis
from gammapy.utils.testing import (
    assert_quantity_allclose,
    mpl_plot_check,
    requires_data,
)


@pytest.fixture(scope="session")
def aeff():
    filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz"
    return EffectiveAreaTable2D.read(filename, hdu="AEFF")


@requires_data()
def test_basic(aeff):
    assert aeff.axes["energy_true"].nbin == 96
    assert aeff.axes["offset"].nbin == 6
    assert aeff.data.shape == (96, 6)

    assert aeff.axes["energy_true"].unit == "TeV"
    assert aeff.axes["offset"].unit == "deg"
    assert aeff.unit == "m2"

    assert_quantity_allclose(aeff.meta["HI_THRES"], 100, rtol=1e-3)
    assert_quantity_allclose(aeff.meta["LO_THRES"], 0.870964, rtol=1e-3)

    test_val = aeff.evaluate(energy_true="14 TeV", offset="0.2 deg")
    assert_allclose(test_val.value, 683177.5, rtol=1e-3)


def test_from_parametrization():
    # Log center of this is 100 GeV
    area_ref = 1.65469579e07 * u.cm**2

    axis = MapAxis.from_energy_edges([80, 125] * u.GeV, name="energy_true")
    area = EffectiveAreaTable2D.from_parametrization(axis, "HESS")

    assert_allclose(area.quantity, area_ref)
    assert area.unit == area_ref.unit

    # Log center of this is 0.1, 2 TeV
    area_ref = [1.65469579e07, 1.46451957e09] * u.cm * u.cm

    axis = MapAxis.from_energy_edges([0.08, 0.125, 32] * u.TeV, name="energy_true")
    area = EffectiveAreaTable2D.from_parametrization(axis, "HESS")
    assert_allclose(area.quantity[:, 0], area_ref)
    assert area.unit == area_ref.unit
    assert area.meta["TELESCOP"] == "HESS"


@requires_data()
def test_plot(aeff):
    with mpl_plot_check():
        aeff.plot()

    with mpl_plot_check():
        aeff.plot_energy_dependence()

    with mpl_plot_check():
        aeff.plot_offset_dependence()


def test_to_table():
    energy_axis_true = MapAxis.from_energy_bounds(
        "1 TeV", "10 TeV", nbin=10, name="energy_true"
    )

    offset_axis = MapAxis.from_bounds(0, 1, nbin=4, name="offset", unit="deg")

    aeff = EffectiveAreaTable2D(
        axes=[energy_axis_true, offset_axis], data=1, unit="cm2"
    )
    hdu = aeff.to_table_hdu()
    assert_equal(hdu.data["ENERG_LO"][0], aeff.axes["energy_true"].edges[:-1].value)
    assert hdu.header["TUNIT1"] == aeff.axes["energy_true"].unit


def test_to_table_is_pointlike():
    energy_axis = MapAxis.from_energy_bounds(
        "1 TeV", "10 TeV", nbin=3, name="energy_true"
    )
    offset_axis = MapAxis.from_bounds(0 * u.deg, 2 * u.deg, nbin=2, name="offset")

    aeff = EffectiveAreaTable2D(
        data=np.ones((3, 2)) * u.m**2, axes=[energy_axis, offset_axis]
    )
    hdu = aeff.to_table_hdu()
    assert "is_pointlike" not in hdu.header


def test_wrong_axis_order():
    energy_axis_true = MapAxis.from_energy_bounds(
        "1 TeV", "10 TeV", nbin=10, name="energy_true"
    )

    offset = np.linspace(0, 1, 4) * u.deg
    offset_axis = MapAxis.from_nodes(offset, name="offset")

    data = np.ones(shape=(offset_axis.nbin, energy_axis_true.nbin))

    with pytest.raises(ValueError):
        EffectiveAreaTable2D(
            axes=[energy_axis_true, offset_axis], data=data, unit="cm2"
        )


def test_wrong_units():
    energy_axis_true = MapAxis.from_energy_bounds(
        "1 TeV", "10 TeV", nbin=10, name="energy_true"
    )

    offset_axis = MapAxis.from_bounds(0 * u.deg, 2 * u.deg, nbin=2, name="offset")

    wrong_unit = u.TeV
    data = np.ones((energy_axis_true.nbin, offset_axis.nbin)) * wrong_unit
    area_test = EffectiveAreaTable2D(axes=[energy_axis_true, offset_axis])

    with pytest.raises(ValueError) as error:
        EffectiveAreaTable2D(data=data, axes=[energy_axis_true, offset_axis])

        assert error.match(
            f"Error: {wrong_unit} is not an allowed unit. {area_test.tag} "
            f"requires {area_test.default_unit} data quantities."
        )


@requires_data("gammapy-data")
def test_aeff2d_pointlike():
    filename = "$GAMMAPY_DATA/joint-crab/dl3/magic/run_05029748_DL3.fits"

    aeff = EffectiveAreaTable2D.read(filename)
    hdu = aeff.to_table_hdu()

    assert aeff.is_pointlike
    assert hdu.header["HDUCLAS3"] == "POINT-LIKE"


def test_eq():
    energy1 = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=2, name="energy_true")

    energy2 = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=3, name="energy_true")

    offset_axis = MapAxis.from_bounds(0 * u.deg, 2 * u.deg, nbin=2, name="offset")

    data1 = np.ones((energy1.nbin, offset_axis.nbin)) * u.cm**2
    data2 = np.ones((energy2.nbin, offset_axis.nbin)) * u.cm**2

    aeff1 = EffectiveAreaTable2D(data=data1, axes=[energy1, offset_axis])
    aeff2 = EffectiveAreaTable2D(data=data2, axes=[energy2, offset_axis])

    assert not aeff1 == aeff2

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from copy import deepcopy
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.irf import Background2D, Background3D
from gammapy.maps import MapAxis
from gammapy.utils.testing import mpl_plot_check, requires_data


@pytest.fixture(scope="session")
def bkg_3d():
    """Example with simple values to test evaluate"""
    energy = [0.1, 10, 1000] * u.TeV
    energy_axis = MapAxis.from_energy_edges(energy)

    fov_lon = [0, 1, 2, 3] * u.deg
    fov_lon_axis = MapAxis.from_edges(fov_lon, name="fov_lon")

    fov_lat = [0, 1, 2, 3] * u.deg
    fov_lat_axis = MapAxis.from_edges(fov_lat, name="fov_lat")

    data = np.ones((2, 3, 3))
    # Axis order is (energy, fov_lon, fov_lat)
    # data.value[1, 0, 0] = 1
    data[1, 1, 1] = 100
    return Background3D(
        axes=[energy_axis, fov_lon_axis, fov_lat_axis], data=data, unit="s-1 GeV-1 sr-1"
    )


@pytest.fixture(scope="session")
def bkg_3d_interp():
    """Example with simple values to test evaluate"""
    energy = np.logspace(-1, 3, 6) * u.TeV
    energy_axis = MapAxis.from_energy_edges(energy)

    fov_lon = [0, 1, 2, 3] * u.deg
    fov_lon_axis = MapAxis.from_edges(fov_lon, name="fov_lon")

    fov_lat = [0, 1, 2, 3] * u.deg
    fov_lat_axis = MapAxis.from_edges(fov_lat, name="fov_lat")

    data = np.ones((5, 3, 3))

    data[-2, :, :] = 0.0
    # clipping of value before last will cause extrapolation problems
    # as found with CTA background IRF

    bkg = Background3D(
        axes=[energy_axis, fov_lon_axis, fov_lat_axis],
        data=data,
        unit="s-1 GeV-1 sr-1",
    )
    return bkg


@requires_data()
def test_background_3d_basics(bkg_3d):
    assert "Background3D" in str(bkg_3d)

    axis = bkg_3d.axes["energy"]
    assert axis.nbin == 2
    assert axis.unit == "TeV"

    axis = bkg_3d.axes["fov_lon"]
    assert axis.nbin == 3
    assert axis.unit == "deg"

    axis = bkg_3d.axes["fov_lat"]
    assert axis.nbin == 3
    assert axis.unit == "deg"

    data = bkg_3d.quantity
    assert data.shape == (2, 3, 3)
    assert data.unit == "s-1 GeV-1 sr-1"

    bkg_2d = bkg_3d.to_2d()
    assert bkg_2d.data.data.shape == (2, 3)

    bkg_3d_new_unit = bkg_3d.to_unit("s-1 MeV-1 sr-1")
    assert_allclose(bkg_3d_new_unit.data[1, 1, 1], 0.1)


def test_background_3d_read_write(tmp_path, bkg_3d):
    bkg_3d.to_table_hdu().writeto(tmp_path / "bkg3d.fits")
    bkg_3d_2 = Background3D.read(tmp_path / "bkg3d.fits")

    axis = bkg_3d_2.axes["energy"]
    assert axis.nbin == 2
    assert axis.unit == "TeV"

    axis = bkg_3d_2.axes["fov_lon"]
    assert axis.nbin == 3
    assert axis.unit == "deg"

    axis = bkg_3d_2.axes["fov_lat"]
    assert axis.nbin == 3
    assert axis.unit == "deg"

    data = bkg_3d_2.quantity
    assert data.shape == (2, 3, 3)
    assert data.unit == "s-1 GeV-1 sr-1"


def test_background_3d_evaluate(bkg_3d):
    # Evaluate at nodes where we put a non-zero value
    res = bkg_3d.evaluate(
        fov_lon=[0.5, 1.5] * u.deg,
        fov_lat=[0.5, 1.5] * u.deg,
        energy=[100, 100] * u.TeV,
    )
    assert_allclose(res.value, [1, 100])
    assert res.shape == (2,)
    assert res.unit == "s-1 GeV-1 sr-1"

    res = bkg_3d.evaluate(
        fov_lon=[1, 0.5] * u.deg,
        fov_lat=[1, 0.5] * u.deg,
        energy=[100, 100] * u.TeV,
    )
    assert_allclose(res.value, [3.162278, 1], rtol=1e-5)

    res = bkg_3d.evaluate(
        fov_lon=[[1, 0.5], [1, 0.5]] * u.deg,
        fov_lat=[[1, 0.5], [1, 0.5]] * u.deg,
        energy=[[1, 1], [100, 100]] * u.TeV,
    )
    assert_allclose(res.value, [[1, 1], [3.162278, 1]], rtol=1e-5)
    assert res.shape == (2, 2)


def test_plot_at_energy(bkg_3d):
    with mpl_plot_check():
        bkg_3d.plot_at_energy(energy=[5] * u.TeV)


def test_background_3d_missing_values(bkg_3d_interp):

    res = bkg_3d_interp.evaluate(
        fov_lon=0.5 * u.deg,
        fov_lat=0.5 * u.deg,
        energy=2000 * u.TeV,
    )
    assert_allclose(res.value, 0.0)

    res = bkg_3d_interp.evaluate(
        fov_lon=0.5 * u.deg,
        fov_lat=0.5 * u.deg,
        energy=999 * u.TeV,
    )
    assert_allclose(res.value, 8.796068e18)
    # without missing value interplation
    # extrapolation within the last bin would give too high value

    bkg_3d_interp.interp_missing_data(axis_name="energy")
    assert np.all(bkg_3d_interp.data != 0)

    bkg_3d_interp.interp_missing_data(axis_name="energy")

    res = bkg_3d_interp.evaluate(
        fov_lon=0.5 * u.deg,
        fov_lat=0.5 * u.deg,
        energy=999 * u.TeV,
    )
    assert_allclose(res.value, 1.0)


def test_background_3d_integrate(bkg_3d):
    # Example has bkg rate = 4 s-1 MeV-1 sr-1 at this node:
    # fov_lon=1.5 deg, fov_lat=1.5 deg, energy=100 TeV

    rate = bkg_3d.integrate_log_log(
        fov_lon=[1.5, 1.5] * u.deg,
        fov_lat=[1.5, 1.5] * u.deg,
        energy=[100, 100 + 2e-6] * u.TeV,
        axis_name="energy",
    )
    assert rate.shape == (1,)

    # Expect approximately `rate * de`
    # with `rate = 4 s-1 sr-1 MeV-1` and `de = 2 MeV`
    assert_allclose(rate.to("s-1 sr-1").value, 0.2, rtol=1e-5)

    rate = bkg_3d.integrate_log_log(
        fov_lon=0.5 * u.deg,
        fov_lat=0.5 * u.deg,
        energy=[1, 100] * u.TeV,
        axis_name="energy",
    )
    assert_allclose(rate.to("s-1 sr-1").value, 99000)

    rate = bkg_3d.integrate_log_log(
        fov_lon=[[1, 0.5], [1, 0.5]] * u.deg,
        fov_lat=[[1, 1], [0.5, 0.5]] * u.deg,
        energy=[[1, 1], [100, 100]] * u.TeV,
        axis_name="energy",
    )
    assert rate.shape == (1, 2)
    assert_allclose(rate.to("s-1 sr-1").value, [[99000.0, 99000.0]], rtol=1e-5)


@requires_data()
def test_background_3d_read():
    filename = (
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    bkg = Background3D.read(filename)
    data = bkg.quantity
    assert bkg.axes.names == ["energy", "fov_lon", "fov_lat"]
    assert data.shape == (21, 36, 36)
    assert data.unit == "s-1 MeV-1 sr-1"


@requires_data()
def test_background_3d_read_gadf():
    filename = "$GAMMAPY_DATA/tests/irf/bkg_3d_full_example.fits"
    bkg = Background3D.read(filename)
    data = bkg.quantity
    assert bkg.axes.names == ["energy", "fov_lon", "fov_lat"]
    assert data.shape == (20, 15, 15)
    assert data.unit == "s-1 MeV-1 sr-1"


def test_bkg_3d_wrong_units():
    energy = [0.1, 10, 1000] * u.TeV
    energy_axis = MapAxis.from_energy_edges(energy)

    fov_lon = [0, 1, 2, 3] * u.deg
    fov_lon_axis = MapAxis.from_edges(fov_lon, name="fov_lon")

    fov_lat = [0, 1, 2, 3] * u.deg
    fov_lat_axis = MapAxis.from_edges(fov_lat, name="fov_lat")

    wrong_unit = u.cm**2 * u.s
    data = np.ones((2, 3, 3)) * wrong_unit
    with pytest.raises(ValueError) as error:
        Background3D(axes=[energy_axis, fov_lon_axis, fov_lat_axis], data=data)
    assert error.match(
        "Error: (.*) is not an allowed unit. (.*) requires (.*) data quantities."
    )


def test_bkg_2d_wrong_units():
    energy = [0.1, 10, 1000] * u.TeV
    energy_axis = MapAxis.from_energy_edges(energy)

    offset_axis = MapAxis.from_edges([0, 1, 2], unit="deg", name="offset")

    wrong_unit = u.cm**2 * u.s
    data = np.ones((energy_axis.nbin, offset_axis.nbin)) * wrong_unit
    bkg2d_test = Background2D(axes=[energy_axis, offset_axis])
    with pytest.raises(ValueError) as error:
        Background2D(axes=[energy_axis, offset_axis], data=data)
        assert error.match(
            f"Error: {wrong_unit} is not an allowed unit. {bkg2d_test.tag}"
            f" requires {bkg2d_test.default_unit} data quantities."
        )


def test_background_2d_read_missing_hducls():
    energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=3)
    offset_axis = MapAxis.from_edges([0, 1, 2], unit="deg", name="offset")

    bkg = Background2D(axes=[energy_axis, offset_axis], unit="s-1 MeV-1 sr-1")

    table = bkg.to_table()
    table.meta.pop("HDUCLAS2")

    bkg = Background2D.from_table(table)

    assert bkg.axes[0].name == "energy"


@pytest.fixture(scope="session")
def bkg_2d():
    """A simple Background2D test case"""
    energy = [0.1, 10, 1000] * u.TeV
    energy_axis = MapAxis.from_energy_edges(energy)

    offset = [0, 1, 2, 3] * u.deg
    offset_axis = MapAxis.from_edges(offset, name="offset")
    data = np.zeros((2, 3))
    data[1, 0] = 2
    data[1, 1] = 4
    return Background2D(
        axes=[energy_axis, offset_axis], data=data, unit="s-1 MeV-1 sr-1"
    )


def test_background_2d_evaluate(bkg_2d):
    # TODO: the test cases here can probably be improved a bit
    # There's some redundancy, and no case exactly at a node in energy

    # Evaluate at log center between nodes in energy
    res = bkg_2d.evaluate(offset=[1, 0.5] * u.deg, energy=[1, 1] * u.TeV)
    assert_allclose(res.value, [0, 0])
    assert res.shape == (2,)
    assert res.unit == "s-1 MeV-1 sr-1"

    res = bkg_2d.evaluate(offset=[1, 0.5] * u.deg, energy=[100, 100] * u.TeV)
    assert_allclose(res.value, [3, 2])
    res = bkg_2d.evaluate(
        offset=[[1, 0.5], [1, 0.5]] * u.deg,
        energy=[[1, 1], [100, 100]] * u.TeV,
    )

    assert_allclose(res.value, [[0, 0], [3, 2]])
    assert res.shape == (2, 2)

    res = bkg_2d.evaluate(offset=[1, 1] * u.deg, energy=[1, 100] * u.TeV)
    assert_allclose(res.value, [0, 3])
    assert res.shape == (2,)


def test_background_2d_read_write(tmp_path, bkg_2d):
    bkg_2d.to_table_hdu().writeto(tmp_path / "tmp.fits")
    bkg_2d_2 = Background2D.read(tmp_path / "tmp.fits")

    axis = bkg_2d_2.axes["energy"]
    assert axis.nbin == 2
    assert axis.unit == "TeV"

    axis = bkg_2d_2.axes["offset"]
    assert axis.nbin == 3
    assert axis.unit == "deg"

    data = bkg_2d_2.data
    assert data.shape == (2, 3)
    assert bkg_2d_2.unit == "s-1 MeV-1 sr-1"


@requires_data()
def test_background_2d_read_gadf():
    filename = "$GAMMAPY_DATA/tests/irf/bkg_2d_full_example.fits"
    bkg = Background2D.read(filename)
    data = bkg.quantity
    assert data.shape == (20, 5)
    assert bkg.axes.names == ["energy", "offset"]
    assert data.unit == "s-1 MeV-1 sr-1"


def test_background_2d_integrate(bkg_2d):
    # TODO: change test case to something better (with known answer)
    # e.g. constant spectrum or power-law.

    rate = bkg_2d.integrate_log_log(
        offset=[1, 0.51] * u.deg, energy=[0.11, 0.5] * u.TeV, axis_name="energy"
    )

    assert rate.shape == (1,)
    assert_allclose(rate.to("s-1 sr-1").value[0], [0, 0])

    rate = bkg_2d.integrate_log_log(
        offset=[1, 0.5] * u.deg, energy=[1, 100] * u.TeV, axis_name="energy"
    )
    assert_allclose(rate.to("s-1 sr-1").value, 0)

    rate = bkg_2d.integrate_log_log(
        offset=[[1, 0.5], [1, 0.5]] * u.deg, energy=[1, 100] * u.TeV, axis_name="energy"
    )
    assert rate.shape == (1, 2)
    assert_allclose(rate.value, [[0, 198]])


def test_to_3d(bkg_2d):
    bkg_3d = bkg_2d.to_3d()
    assert bkg_3d.data.shape == (2, 6, 6)
    assert_allclose(bkg_3d.data[1, 1, 1], 1.51, rtol=0.1)

    # assert you get back same after goint to 2d
    # need high rtol due to interpolation effects?
    b2 = bkg_3d.to_2d()
    assert_allclose(bkg_2d.data, b2.data, rtol=0.2)
    assert b2.unit == bkg_2d.unit


def test_plot(bkg_2d):
    with mpl_plot_check():
        bkg_2d.plot()

    with mpl_plot_check():
        bkg_2d.plot_energy_dependence()

    with mpl_plot_check():
        bkg_2d.plot_offset_dependence()

    with mpl_plot_check():
        bkg_2d.plot_spectrum()

    with mpl_plot_check():
        bkg_2d.peek()

    with mpl_plot_check():
        bkg_2d.plot_at_energy(energy=[1.0, 5.0] * u.TeV)


def test_eq(bkg_2d):
    bkg1 = deepcopy(bkg_2d)
    assert bkg1 == bkg_2d

    bkg1.data[0][0] = 10
    assert not bkg1 == bkg_2d

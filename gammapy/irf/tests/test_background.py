# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.irf import Background2D, Background3D
from gammapy.maps import MapAxis
from gammapy.utils.testing import mpl_plot_check, requires_data, requires_dependency


@pytest.fixture(scope="session")
def bkg_3d():
    """Example with simple values to test evaluate"""
    energy = [0.1, 10, 1000] * u.TeV
    energy_axis = MapAxis.from_energy_edges(energy)

    fov_lon = [0, 1, 2, 3] * u.deg
    fov_lon_axis = MapAxis.from_edges(fov_lon, name="fov_lon")

    fov_lat = [0, 1, 2, 3] * u.deg
    fov_lat_axis = MapAxis.from_edges(fov_lat, name="fov_lat")

    data = np.ones((2, 3, 3)) * u.Unit("s-1 GeV-1 sr-1")
    # Axis order is (energy, fov_lon, fov_lat)
    # data.value[1, 0, 0] = 1
    data.value[1, 1, 1] = 100
    return Background3D(
        energy_axis=energy_axis,
        fov_lon_axis=fov_lon_axis,
        fov_lat_axis=fov_lat_axis,
        data=data,
    )


@requires_data()
def test_background_3d_basics(bkg_3d):
    assert "NDDataArray summary info" in str(bkg_3d.data)

    axis = bkg_3d.data.axes["energy"]
    assert axis.nbin == 2
    assert axis.unit == "TeV"

    axis = bkg_3d.data.axes["fov_lon"]
    assert axis.nbin == 3
    assert axis.unit == "deg"

    axis = bkg_3d.data.axes["fov_lat"]
    assert axis.nbin == 3
    assert axis.unit == "deg"

    data = bkg_3d.data.data
    assert data.shape == (2, 3, 3)
    assert data.unit == "s-1 GeV-1 sr-1"

    bkg_2d = bkg_3d.to_2d()
    assert bkg_2d.data.data.shape == (2, 3)


def test_background_3d_read_write(tmp_path, bkg_3d):
    bkg_3d.to_table_hdu().writeto(tmp_path / "bkg3d.fits")
    bkg_3d_2 = Background3D.read(tmp_path / "bkg3d.fits")

    axis = bkg_3d_2.data.axes["energy"]
    assert axis.nbin == 2
    assert axis.unit == "TeV"

    axis = bkg_3d_2.data.axes["fov_lon"]
    assert axis.nbin == 3
    assert axis.unit == "deg"

    axis = bkg_3d_2.data.axes["fov_lat"]
    assert axis.nbin == 3
    assert axis.unit == "deg"

    data = bkg_3d_2.data.data
    assert data.shape == (2, 3, 3)
    assert data.unit == "s-1 GeV-1 sr-1"


def test_background_3d_evaluate(bkg_3d):
    # Evaluate at nodes where we put a non-zero value
    res = bkg_3d.evaluate(
        fov_lon=[0.5, 1.5] * u.deg,
        fov_lat=[0.5, 1.5] * u.deg,
        energy_reco=[100, 100] * u.TeV,
    )
    assert_allclose(res.value, [1, 100])
    assert res.shape == (2,)
    assert res.unit == "s-1 GeV-1 sr-1"

    res = bkg_3d.evaluate(
        fov_lon=[1, 0.5] * u.deg,
        fov_lat=[1, 0.5] * u.deg,
        energy_reco=[100, 100] * u.TeV,
    )
    assert_allclose(res.value, [3.162278, 1], rtol=1e-5)

    res = bkg_3d.evaluate(
        fov_lon=[[1, 0.5], [1, 0.5]] * u.deg,
        fov_lat=[[1, 0.5], [1, 0.5]] * u.deg,
        energy_reco=[[1, 1], [100, 100]] * u.TeV,
    )
    assert_allclose(res.value, [[1, 1], [3.162278, 1]], rtol=1e-5)
    assert res.shape == (2, 2)


def test_background_3d_integrate(bkg_3d):
    # Example has bkg rate = 4 s-1 MeV-1 sr-1 at this node:
    # fov_lon=1.5 deg, fov_lat=1.5 deg, energy=100 TeV

    rate = bkg_3d.evaluate_integrate(
        fov_lon=[1.5, 1.5] * u.deg,
        fov_lat=[1.5, 1.5] * u.deg,
        energy_reco=[100, 100 + 2e-6] * u.TeV,
    )
    assert rate.shape == (1,)

    # Expect approximately `rate * de`
    # with `rate = 4 s-1 sr-1 MeV-1` and `de = 2 MeV`
    assert_allclose(rate.to("s-1 sr-1").value, 0.2, rtol=1e-5)

    rate = bkg_3d.evaluate_integrate(
        fov_lon=0.5 * u.deg, fov_lat=0.5 * u.deg, energy_reco=[1, 100] * u.TeV
    )
    assert_allclose(rate.to("s-1 sr-1").value, 99000)

    rate = bkg_3d.evaluate_integrate(
        fov_lon=[[1, 0.5], [1, 0.5]] * u.deg,
        fov_lat=[[1, 1], [0.5, 0.5]] * u.deg,
        energy_reco=[[1, 1], [100, 100]] * u.TeV,
    )
    assert rate.shape == (1, 2)
    assert_allclose(rate.to("s-1 sr-1").value, [[99000.0, 99000.0]], rtol=1e-5)


@requires_data()
def test_background_3D_read():
    filename = (
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    bkg = Background3D.read(filename)
    data = bkg.data.data
    assert bkg.data.axes.names == ["energy", "fov_lon", "fov_lat"]
    assert data.shape == (21, 36, 36)
    assert data.unit == "s-1 MeV-1 sr-1"


@requires_data()
def test_background_3D_read_gadf():
    filename = "$GAMMAPY_DATA/tests/irf/bkg_3d_full_example.fits"
    bkg = Background3D.read(filename)
    data = bkg.data.data
    assert bkg.data.axes.names == ["energy", "fov_lon", "fov_lat"]
    assert data.shape == (20, 15, 15)
    assert data.unit == "s-1 MeV-1 sr-1"


@requires_dependency("matplotlib")
def test_plot(bkg_2d):
    with mpl_plot_check():
        bkg_2d.peek()


@pytest.fixture(scope="session")
def bkg_2d():
    """A simple Background2D test case"""
    energy = [0.1, 10, 1000] * u.TeV
    energy_axis = MapAxis.from_energy_edges(energy)

    offset = [0, 1, 2, 3] * u.deg
    offset_axis = MapAxis.from_edges(offset, name="offset")
    data = np.zeros((2, 3)) * u.Unit("s-1 MeV-1 sr-1")
    data.value[1, 0] = 2
    data.value[1, 1] = 4
    return Background2D(energy_axis=energy_axis, offset_axis=offset_axis, data=data,)


def test_background_2d_evaluate(bkg_2d):
    # TODO: the test cases here can probably be improved a bit
    # There's some redundancy, and no case exactly at a node in energy

    # Evaluate at log center between nodes in energy
    res = bkg_2d.evaluate(
        fov_lon=[1, 0.5] * u.deg, fov_lat=0 * u.deg, energy_reco=[1, 1] * u.TeV
    )
    assert_allclose(res.value, [0, 0])
    assert res.shape == (2,)
    assert res.unit == "s-1 MeV-1 sr-1"

    res = bkg_2d.evaluate(
        fov_lon=[1, 0.5] * u.deg, fov_lat=0 * u.deg, energy_reco=[100, 100] * u.TeV
    )
    assert_allclose(res.value, [3, 2])
    res = bkg_2d.evaluate(
        fov_lon=[[1, 0.5], [1, 0.5]] * u.deg,
        fov_lat=0 * u.deg,
        energy_reco=[[1, 1], [100, 100]] * u.TeV,
    )

    assert_allclose(res.value, [[0, 0], [3, 2]])
    assert res.shape == (2, 2)

    res = bkg_2d.evaluate(
        fov_lon=[1, 1] * u.deg, fov_lat=0 * u.deg, energy_reco=[1, 100] * u.TeV
    )
    assert_allclose(res.value, [0, 3])
    assert res.shape == (2,)


def test_background_2d_read_write(tmp_path, bkg_2d):
    bkg_2d.to_table_hdu().writeto(tmp_path / "tmp.fits")
    bkg_2d_2 = Background2D.read(tmp_path / "tmp.fits")

    axis = bkg_2d_2.data.axes["energy"]
    assert axis.nbin == 2
    assert axis.unit == "TeV"

    axis = bkg_2d_2.data.axes["offset"]
    assert axis.nbin == 3
    assert axis.unit == "deg"

    data = bkg_2d_2.data.data
    assert data.shape == (2, 3)
    assert data.unit == "s-1 MeV-1 sr-1"


@requires_data()
def test_background_2D_read_gadf():
    filename = "$GAMMAPY_DATA/tests/irf/bkg_2d_full_example.fits"
    bkg = Background2D.read(filename)
    data = bkg.data.data
    assert data.shape == (20, 5)
    assert bkg.data.axes.names == ["energy", "offset"]
    assert data.unit == "s-1 MeV-1 sr-1"


def test_background_2d_integrate(bkg_2d):
    # TODO: change test case to something better (with known answer)
    # e.g. constant spectrum or power-law.

    rate = bkg_2d.evaluate_integrate(
        fov_lon=[1, 0.5] * u.deg, fov_lat=[0, 0] * u.deg, energy_reco=[0.1, 0.5] * u.TeV
    )

    assert rate.shape == (1,)
    assert_allclose(rate.to("s-1 sr-1").value[0], [0, 0])

    rate = bkg_2d.evaluate_integrate(
        fov_lon=[1, 0.5] * u.deg, fov_lat=[0, 0] * u.deg, energy_reco=[1, 100] * u.TeV
    )
    assert_allclose(rate.to("s-1 sr-1").value, 0)

    rate = bkg_2d.evaluate_integrate(
        fov_lon=[[1, 0.5], [1, 0.5]] * u.deg,
        fov_lat=0 * u.deg,
        energy_reco=[1, 100] * u.TeV,
    )
    assert rate.shape == (1, 2)
    assert_allclose(rate.value, [[0, 198]])


@requires_dependency("matplotlib")
def test_plot(bkg_2d):
    with mpl_plot_check():
        bkg_2d.plot()

    with mpl_plot_check():
        bkg_2d.plot_energy_dependence()

    with mpl_plot_check():
        bkg_2d.plot_offset_dependence()

    with mpl_plot_check():
        bkg_2d.plot_spectrum()

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
import scipy.special
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.irf import IRF
from gammapy.makers.utils import make_edisp_kernel_map, make_map_exposure_true_energy
from gammapy.maps import MapAxes, MapAxis, WcsGeom


class EffectiveArea3D(IRF):
    tag = "aeff_3d"
    required_axes = ["energy_true", "fov_lon", "fov_lat"]
    default_unit = u.m**2


@pytest.fixture
def aeff_3d():
    energy_axis = MapAxis.from_energy_edges(
        [0.1, 0.3, 1.0, 3.0, 10.0] * u.TeV, name="energy_true"
    )
    fov_lon_axis = MapAxis.from_edges([-1.5, -0.5, 0.5, 1.5] * u.deg, name="fov_lon")
    fov_lat_axis = MapAxis.from_edges([-1.5, -0.5, 0.5, 1.5] * u.deg, name="fov_lat")

    data = np.ones((4, 3, 3))
    for i in range(1, 4):
        data[i] = data[i - 1] * 1.5

    return EffectiveArea3D(
        [energy_axis, fov_lon_axis, fov_lat_axis], data=data, unit=u.m**2
    )


class EnergyDispersion3D(IRF):
    tag = "edisp_3d"
    required_axes = ["energy_true", "migra", "fov_lon", "fov_lat"]
    default_unit = u.one

    @classmethod
    def from_gauss(
        cls,
        energy_axis_true,
        migra_axis,
        fov_lon_axis,
        fov_lat_axis,
        bias,
        sigma,
        pdf_threshold=1e-6,
    ):
        axes = MapAxes([energy_axis_true, migra_axis, fov_lon_axis, fov_lat_axis])
        coords = axes.get_coord(mode="edges", axis_name="migra")

        migra_min = coords["migra"][:, :-1, :]
        migra_max = coords["migra"][:, 1:, :]

        # Analytical formula for integral of Gaussian
        s = np.sqrt(2) * sigma
        t1 = (migra_max - 1 - bias) / s
        t2 = (migra_min - 1 - bias) / s
        pdf = (scipy.special.erf(t1) - scipy.special.erf(t2)) / 2
        pdf = pdf / (migra_max - migra_min)

        r1 = np.rollaxis(pdf, -1, 1)
        r2 = np.rollaxis(r1, 0, -1)
        data = r2 * np.ones(axes.shape)

        data[data < pdf_threshold] = 0

        return cls(
            axes=axes,
            data=data.value,
        )


@pytest.fixture
def edisp_3d():
    energy_axis_true = MapAxis.from_energy_bounds(
        "0.1 TeV", "100 TeV", nbin=5, name="energy_true"
    )

    migra_axis = MapAxis.from_bounds(0, 4, nbin=10, node_type="edges", name="migra")

    fov_lon_axis = MapAxis.from_edges([-1.5, -0.5, 0.5, 1.5] * u.deg, name="fov_lon")
    fov_lat_axis = MapAxis.from_edges([-1.5, -0.5, 0.5, 1.5] * u.deg, name="fov_lat")

    energy_true = energy_axis_true.edges[:-1]
    sigma = 0.15 / (energy_true / (1 * u.TeV)).value ** 0.3
    bias = 1e-3 * (energy_true - 1 * u.TeV).value

    return EnergyDispersion3D.from_gauss(
        energy_axis_true=energy_axis_true,
        migra_axis=migra_axis,
        fov_lon_axis=fov_lon_axis,
        fov_lat_axis=fov_lat_axis,
        bias=bias,
        sigma=sigma,
    )


def test_aeff_3d(aeff_3d):
    res = aeff_3d.evaluate(
        fov_lon=[-0.5, 0.8] * u.deg,
        fov_lat=[-0.5, 1.0] * u.deg,
        energy_true=[0.2, 8.0] * u.TeV,
    )
    assert_allclose(res.data, [1.06246937, 3.74519106], rtol=1e-5)

    axis = MapAxis.from_energy_bounds(0.1 * u.TeV, 10 * u.TeV, 6, name="energy_true")
    pointing = SkyCoord(2, 1, unit="deg")
    geom = WcsGeom.create(npix=(4, 3), binsz=2, axes=[axis], skydir=pointing)

    exposure_map = make_map_exposure_true_energy(
        pointing=pointing, livetime="42 h", aeff=aeff_3d, geom=geom
    )
    assert_allclose(
        exposure_map.data[3][1][1:3], [323894.44971479, 323894.44971479], rtol=1e-5
    )


def test_edisp_3d(edisp_3d):
    energy = [1, 2] * u.TeV
    migra = np.array([0.98, 0.97, 0.7])
    fov_lon = [0.1, 1.5] * u.deg
    fov_lat = [0.0, 0.3] * u.deg

    eval = edisp_3d.evaluate(
        energy_true=energy.reshape(-1, 1, 1, 1),
        migra=migra.reshape(1, -1, 1, 1),
        fov_lon=fov_lon.reshape(1, 1, -1, 1),
        fov_lat=fov_lat.reshape(1, 1, 1, -1),
    )

    assert_allclose(
        eval[0][2], [[0.71181427, 0.71181427], [0.71181427, 0.71181427]], rtol=1e-3
    )
    etrue = MapAxis.from_energy_bounds(0.5, 2, 6, unit="TeV", name="energy_true")
    ereco = MapAxis.from_energy_bounds(0.5, 2, 3, unit="TeV", name="energy")
    pointing = SkyCoord(2, 1, unit="deg")
    geom = WcsGeom.create(10, binsz=0.5, axes=[ereco, etrue], skydir=pointing)

    edispmap = make_edisp_kernel_map(edisp_3d, pointing, geom)
    assert_allclose(edispmap.edisp_map.data[3][1][5][5], 0.623001, rtol=1e-3)

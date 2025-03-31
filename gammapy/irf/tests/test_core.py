# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.irf.core import IRF, FoVAlignment
from gammapy.maps import MapAxis, WcsNDMap
from gammapy.irf import EDispKernelMap


class MyCustomIRF(IRF):
    tag = "myirf"
    required_axes = ["energy", "offset"]
    default_unit = u.deg


def test_irf_init_quantity():
    energy_axis = MapAxis.from_energy_bounds(10, 100, 10, unit="TeV", name="energy")
    offset_axis = MapAxis.from_bounds(0, 2.5, 5, unit="deg", name="offset")
    data = np.full((10, 5), 1)

    irf = MyCustomIRF(axes=[energy_axis, offset_axis], data=data, unit=u.deg)
    irf2 = MyCustomIRF(axes=[energy_axis, offset_axis], data=data * u.deg)

    assert np.all(irf.quantity == irf2.quantity)


def test_immutable():
    energy_axis = MapAxis.from_energy_bounds(10, 100, 10, unit="TeV", name="energy")
    offset_axis = MapAxis.from_bounds(0, 2.5, 5, unit="deg", name="offset")
    data = np.full((10, 5), 1) * u.deg
    test_irf = MyCustomIRF(
        axes=[energy_axis, offset_axis],
        data=data,
        is_pointlike=False,
        fov_alignment=FoVAlignment.RADEC,
    )

    with pytest.raises(AttributeError):
        test_irf.is_pointlike = True

    with pytest.raises(AttributeError):
        test_irf.fov_alignment = FoVAlignment.ALTAZ


def test_slice_by_idx():
    energy_axis = MapAxis.from_energy_bounds(10, 100, 10, unit="TeV", name="energy")
    offset_axis = MapAxis.from_bounds(0, 2.5, 5, unit="deg", name="offset")
    data = np.full((10, 5), 1)

    irf = MyCustomIRF(axes=[energy_axis, offset_axis], data=data, unit=u.deg)

    irf_sliced = irf.slice_by_idx({"energy": slice(3, 7)})
    assert irf_sliced.data.shape == (4, 5)
    assert irf_sliced.axes["energy"].nbin == 4

    irf_sliced = irf.slice_by_idx({"offset": slice(3, 5)})
    assert irf_sliced.data.shape == (10, 2)
    assert irf_sliced.axes["offset"].nbin == 2

    irf_sliced = irf.slice_by_idx({"energy": slice(3, 7), "offset": slice(3, 5)})
    assert irf_sliced.data.shape == (4, 2)
    assert irf_sliced.axes["offset"].nbin == 2
    assert irf_sliced.axes["energy"].nbin == 4

    with pytest.raises(ValueError) as exc_info:
        _ = irf.slice_by_idx({"energy": 3, "offset": 7})

    assert (
        str(exc_info.value)
        == "Integer indexing not supported, got {'energy': 3, 'offset': 7}"
    )


def test_cum_sum():
    energy_axis = MapAxis.from_energy_bounds(10, 100, 10, unit="TeV", name="energy")
    offset_axis = MapAxis.from_bounds(0, 2.5, 1, unit="deg", name="offset")

    data = np.full((10, 1), 1)

    irf = MyCustomIRF(axes=[energy_axis, offset_axis], data=data, unit="")
    cumsum = irf.cumsum(axis_name="offset")

    assert cumsum.unit == u.Unit("deg^2")
    assert cumsum.data[0, 0] == 2.5**2 * np.pi


def test_irfmap_downsample():
    energy_axis = MapAxis.from_energy_bounds(10, 100, 10, unit="TeV", name="energy")
    energy_true_axis = MapAxis.from_bounds(1e-1, 10, 40, unit="TeV", name="energy_true")
    m = WcsNDMap.create(npix=(200, 200), axes=[energy_axis, energy_true_axis])
    expmap = WcsNDMap.from_geom(m.geom.drop("energy"), unit="cm^2 s")
    m.data = np.random.rand(*m.data.shape)
    expmap.data = np.random.rand(*expmap.data.shape)

    weights = m.copy()
    weights.data = np.tile(
        np.array([1.0, 2.0]),
        (m.data.shape[0], m.data.shape[1], m.data.shape[2], m.data.shape[3] // 2),
    )

    irf = EDispKernelMap(m, expmap)
    irf2 = EDispKernelMap(m, None)

    # test spatial downsampling without weights
    irf3 = irf.downsample(2)

    assert irf3.edisp_map.unit == irf.edisp_map.unit
    assert irf3.edisp_map.geom.npix[0] == irf.edisp_map.geom.npix[0] / 2
    assert_allclose(
        np.mean(irf.edisp_map.data[:, :, 0:2, 0:2], axis=(2, 3)),
        irf3.edisp_map.data[:, :, 0, 0],
    )

    assert irf3.exposure_map.unit == irf.exposure_map.unit
    assert irf3.exposure_map.geom.npix[0] == irf.exposure_map.geom.npix[0] / 2
    assert_allclose(
        np.mean(irf.exposure_map.data[:, 0:2, 0:2], axis=(-1, -2)),
        irf3.exposure_map.data[:, 0, 0],
    )

    # test spatial downsampling without exposure map
    irf2.downsample(2)

    # test energy downsampling with weights
    irf5 = irf.downsample(2, axis_name="energy", weights=weights)
    assert irf5.edisp_map.data.shape[1] == irf2.edisp_map.data.shape[1] / 2
    assert_allclose(
        np.sum(
            irf2.edisp_map.data[:, 0:2, :, :] * weights.data[:, 0:2, :, :], axis=(1)
        ),
        irf5.edisp_map.data[:, 0, :, :],
    )
    assert irf5.exposure_map == irf.exposure_map

    # test spatial downsampling with weights
    irf6 = irf.downsample(2, weights=weights)
    assert_allclose(
        np.sum(
            irf.edisp_map.data[:, :, 0:2, 0:2] * weights.data[:, :, 0:2, 0:2],
            axis=(2, 3),
        )
        / np.sum(weights.data[:, :, 0:2, 0:2], axis=(2, 3)),
        irf6.edisp_map.data[:, :, 0, 0],
    )

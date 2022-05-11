# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.units import Unit
from gammapy.irf import (
    EDispKernel,
    EDispKernelMap,
    EDispMap,
    EffectiveAreaTable2D,
    EnergyDispersion2D,
)
from gammapy.makers.utils import make_edisp_map, make_map_exposure_true_energy
from gammapy.maps import MapAxis, MapCoord, RegionGeom, WcsGeom
from gammapy.utils.testing import mpl_plot_check


def fake_aeff2d(area=1e6 * u.m**2):
    offsets = np.array((0.0, 1.0, 2.0, 3.0)) * u.deg

    energy_axis_true = MapAxis.from_energy_bounds(
        "0.1 TeV", "10 TeV", nbin=4, name="energy_true"
    )

    offset_axis = MapAxis.from_edges(offsets, name="offset")
    return EffectiveAreaTable2D(
        axes=[energy_axis_true, offset_axis], data=area.value, unit=area.unit
    )


def make_edisp_map_test():
    pointing = SkyCoord(0, 0, unit="deg")

    energy_axis_true = MapAxis.from_energy_edges(
        energy_edges=[0.2, 0.7, 1.5, 2.0, 10.0] * u.TeV,
        name="energy_true",
    )

    migra_axis = MapAxis(nodes=np.linspace(0.0, 3.0, 51), unit="", name="migra")

    offset_axis = MapAxis.from_nodes([0.0, 1.0, 2.0, 3.0] * u.deg, name="offset")

    edisp2d = EnergyDispersion2D.from_gauss(
        energy_axis_true=energy_axis_true,
        migra_axis=migra_axis,
        offset_axis=offset_axis,
        bias=0,
        sigma=0.2,
    )

    geom = WcsGeom.create(
        skydir=pointing, binsz=1.0, width=5.0, axes=[migra_axis, energy_axis_true]
    )

    aeff2d = fake_aeff2d()
    exposure_geom = geom.squash(axis_name="migra")
    exposure_map = make_map_exposure_true_energy(pointing, "1 h", aeff2d, exposure_geom)

    return make_edisp_map(edisp2d, pointing, geom, exposure_map)


def test_make_edisp_map():
    energy_axis = MapAxis(
        nodes=[0.2, 0.7, 1.5, 2.0, 10.0],
        unit="TeV",
        name="energy_true",
        node_type="edges",
        interp="log",
    )
    migra_axis = MapAxis(nodes=np.linspace(0.0, 3.0, 51), unit="", name="migra")

    edmap = make_edisp_map_test()

    assert edmap.edisp_map.geom.axes[0] == migra_axis
    assert edmap.edisp_map.geom.axes[1] == energy_axis
    assert edmap.edisp_map.unit == Unit("")
    assert edmap.edisp_map.data.shape == (4, 50, 5, 5)


def test_edisp_map_to_from_hdulist():
    edmap = make_edisp_map_test()
    hdulist = edmap.to_hdulist()
    assert "EDISP" in hdulist
    assert "EDISP_BANDS" in hdulist
    assert "EDISP_EXPOSURE" in hdulist
    assert "EDISP_EXPOSURE_BANDS" in hdulist

    new_edmap = EDispMap.from_hdulist(hdulist)
    assert_allclose(edmap.edisp_map.data, new_edmap.edisp_map.data)
    assert new_edmap.edisp_map.geom == edmap.edisp_map.geom
    assert new_edmap.exposure_map.geom == edmap.exposure_map.geom


def test_edisp_map_read_write(tmp_path):
    edisp_map = make_edisp_map_test()

    edisp_map.write(tmp_path / "tmp.fits")
    new_edmap = EDispMap.read(tmp_path / "tmp.fits")

    assert_allclose(edisp_map.edisp_map.quantity, new_edmap.edisp_map.quantity)


def test_edisp_map_to_energydispersion():
    edmap = make_edisp_map_test()

    position = SkyCoord(0, 0, unit="deg")
    energy_axis = MapAxis.from_edges(np.logspace(-0.3, 0.2, 200) * u.TeV, name="energy")

    edisp = edmap.get_edisp_kernel(position=position, energy_axis=energy_axis)
    # Note that the bias and resolution are rather poorly evaluated on an EnergyDispersion object
    assert_allclose(edisp.get_bias(energy_true=1.0 * u.TeV), 0.0, atol=3e-2)
    assert_allclose(edisp.get_resolution(energy_true=1.0 * u.TeV), 0.2, atol=3e-2)


def test_edisp_map_from_geom_error():
    energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=3)
    energy_axis_true = MapAxis.from_energy_bounds(
        "1 TeV", "10 TeV", nbin=3, name="energy_true"
    )

    geom = WcsGeom.create(npix=(1, 1), axes=[energy_axis_true, energy_axis])

    with pytest.raises(ValueError):
        EDispKernelMap.from_geom(geom=geom)


def test_edisp_map_stacking():
    edmap1 = make_edisp_map_test()
    edmap2 = make_edisp_map_test()
    edmap2.exposure_map.quantity *= 2

    edmap_stack = edmap1.copy()
    edmap_stack.stack(edmap2)
    assert_allclose(edmap_stack.edisp_map.data, edmap1.edisp_map.data)
    assert_allclose(edmap_stack.exposure_map.data, edmap1.exposure_map.data * 3)


def test_sample_coord():
    edisp_map = make_edisp_map_test()

    coords = MapCoord(
        {"lon": [0, 0] * u.deg, "lat": [0, 0.5] * u.deg, "energy_true": [1, 3] * u.TeV},
        frame="icrs",
    )

    coords_corrected = edisp_map.sample_coord(map_coord=coords)

    assert len(coords_corrected["energy"]) == 2
    assert coords_corrected["energy"].unit == "TeV"
    assert_allclose(coords_corrected["energy"].value, [1.024664, 3.34484], rtol=1e-5)


@pytest.mark.parametrize("position", ["0d 0d", "180d 0d", "0d 90d", "180d -90d"])
def test_edisp_from_diagonal_response(position):
    position = SkyCoord(position)
    energy_axis_true = MapAxis.from_energy_bounds(
        "0.3 TeV", "10 TeV", nbin=31, name="energy_true"
    )
    energy_axis = MapAxis.from_energy_bounds(
        "0.3 TeV", "10 TeV", nbin=31, name="energy"
    )

    edisp_map = EDispMap.from_diagonal_response(energy_axis_true)
    edisp_kernel = edisp_map.get_edisp_kernel(
        position=position, energy_axis=energy_axis
    )

    sum_kernel = np.sum(edisp_kernel.data, axis=1)

    # We exclude the first and last bin, where there is no
    # e_reco to contribute to
    assert_allclose(sum_kernel[1:-1], 1)


def test_edisp_map_to_edisp_kernel_map():
    energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=5)

    energy_axis_true = MapAxis.from_energy_bounds(
        "0.3 TeV", "30 TeV", nbin=10, per_decade=True, name="energy_true"
    )
    migra_axis = MapAxis(nodes=np.linspace(0.0, 3.0, 51), unit="", name="migra")

    edisp_map = EDispMap.from_diagonal_response(energy_axis_true, migra_axis)

    edisp_kernel_map = edisp_map.to_edisp_kernel_map(energy_axis)
    position = SkyCoord(0, 0, unit="deg")
    kernel = edisp_kernel_map.get_edisp_kernel(position=position)

    assert edisp_kernel_map.exposure_map.geom.axes[0].name == "energy"
    actual = kernel.pdf_matrix.sum(axis=0)
    assert_allclose(actual, 2.0)


def test_edisp_kernel_map_stack():
    energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=5)

    energy_axis_true = MapAxis.from_energy_bounds(
        "0.3 TeV", "30 TeV", nbin=10, per_decade=True, name="energy_true"
    )

    edisp_1 = EDispKernelMap.from_diagonal_response(
        energy_axis=energy_axis, energy_axis_true=energy_axis_true
    )
    edisp_1.exposure_map.data += 1

    edisp_2 = EDispKernelMap.from_diagonal_response(
        energy_axis=energy_axis, energy_axis_true=energy_axis_true
    )
    edisp_2.exposure_map.data += 2

    geom = edisp_1.edisp_map.geom
    weights = geom.energy_mask(energy_min=2 * u.TeV)
    edisp_1.stack(edisp_2, weights=weights)

    position = SkyCoord(0, 0, unit="deg")
    kernel = edisp_1.get_edisp_kernel(position=position)

    actual = kernel.pdf_matrix.sum(axis=0)
    exposure = edisp_1.exposure_map.data[:, 0, 0, 0]

    assert_allclose(actual, [2.0 / 3.0, 2.0 / 3.0, 2.0, 2.0, 2.0])
    assert_allclose(exposure, 3.0)


def test__incorrect_edisp_kernel_map_stack():
    energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=5)

    energy_axis_true = MapAxis.from_energy_bounds(
        "0.3 TeV", "30 TeV", nbin=10, per_decade=True, name="energy_true"
    )

    edisp_1 = EDispKernelMap.from_diagonal_response(
        energy_axis=energy_axis, energy_axis_true=energy_axis_true
    )
    edisp_1.exposure_map.data += 1

    edisp_2 = EDispKernelMap.from_diagonal_response(
        energy_axis=energy_axis, energy_axis_true=energy_axis_true
    )
    edisp_2.exposure_map = None

    with pytest.raises(ValueError) as except_info:
        edisp_1.stack(edisp_2)
    assert except_info.match("Missing exposure map for EDispKernelMap.stack")


def test_edispkernel_from_diagonal_response():
    energy_axis_true = MapAxis.from_energy_bounds(
        "0.3 TeV", "10 TeV", nbin=11, name="energy_true"
    )
    energy_axis = MapAxis.from_energy_bounds(
        "0.3 TeV", "10 TeV", nbin=11, name="energy"
    )

    geom = RegionGeom.create("fk5;circle(0, 0, 10)")
    region_edisp = EDispKernelMap.from_diagonal_response(
        energy_axis, energy_axis_true, geom=geom
    )
    sum_kernel = np.sum(region_edisp.edisp_map.data[..., 0, 0], axis=1)

    # We exclude the first and last bin, where there is no
    # e_reco to contribute to
    assert_allclose(sum_kernel[1:-1], 1)


def test_edispkernel_from_1d():
    energy_axis_true = MapAxis.from_energy_bounds(
        "0.5 TeV", "5 TeV", nbin=31, name="energy_true"
    )
    energy_axis = MapAxis.from_energy_bounds(
        "0.1 TeV", "10 TeV", nbin=11, name="energy"
    )

    edisp = EDispKernel.from_gauss(energy_axis_true, energy_axis, 0.1, 0.0)

    geom = RegionGeom.create("fk5;circle(0, 0, 10)")
    region_edisp = EDispKernelMap.from_edisp_kernel(edisp, geom=geom)

    sum_kernel = np.sum(region_edisp.edisp_map.data[..., 0, 0], axis=1)
    assert_allclose(sum_kernel, 1, rtol=1e-5)

    allsky_edisp = EDispKernelMap.from_edisp_kernel(edisp)

    sum_kernel = np.sum(allsky_edisp.edisp_map.data[..., 0, 0], axis=1)
    assert allsky_edisp.edisp_map.data.shape == (31, 11, 1, 2)
    assert_allclose(sum_kernel, 1, rtol=1e-5)


def test_edisp_kernel_map_to_image():
    e_reco = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=3)
    e_true = MapAxis.from_energy_bounds(
        "0.08 TeV", "20 TeV", nbin=5, name="energy_true"
    )
    edisp = EDispKernelMap.from_diagonal_response(e_reco, e_true)
    im = edisp.to_image()

    assert im.edisp_map.data.shape == (5, 1, 1, 2)
    assert_allclose(im.edisp_map.data[0, 0, 0, 0], 0.87605894, rtol=1e-5)


def test_edisp_kernel_map_resample_axis():
    e_reco = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=4)
    e_true = MapAxis.from_energy_bounds(
        "0.08 TeV", "20 TeV", nbin=10, name="energy_true"
    )
    edisp = EDispKernelMap.from_diagonal_response(e_reco, e_true)

    e_reco = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=2)
    im = edisp.resample_energy_axis(energy_axis=e_reco)

    res = np.sum(im.edisp_map.data[4, :, 0, 0])

    assert im.edisp_map.data.shape == (10, 2, 1, 2)
    assert_allclose(res, 1.0, rtol=1e-5)


def test_peek():
    e_reco = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=3)
    e_true = MapAxis.from_energy_bounds(
        "0.08 TeV", "20 TeV", nbin=5, name="energy_true"
    )
    edisp = EDispKernelMap.from_diagonal_response(e_reco, e_true)
    with mpl_plot_check():
        edisp.peek()
    edisp = EDispMap.from_diagonal_response(e_true)
    with mpl_plot_check():
        edisp.peek()

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from regions import CircleSkyRegion
from gammapy.irf import PSFKernel, PSFMap
from gammapy.maps import HpxGeom, HpxMap, HpxNDMap, Map, MapAxis, WcsGeom
from gammapy.maps.hpx.io import HpxConv
from gammapy.maps.io import find_bintable_hdu
from gammapy.utils.testing import mpl_plot_check, requires_data

pytest.importorskip("healpy")

axes1 = [MapAxis(np.logspace(0.0, 3.0, 3), interp="log")]

hpx_test_allsky_geoms = [
    (8, False, "galactic", None, None),
    (8, False, "galactic", None, axes1),
    ([4, 8], False, "galactic", None, axes1),
]

hpx_test_partialsky_geoms = [
    ([4, 8], False, "galactic", "DISK(110.,75.,30.)", axes1),
    (8, False, "galactic", "DISK(110.,75.,10.)", [MapAxis(np.logspace(0.0, 3.0, 4))]),
    (
        8,
        False,
        "galactic",
        "DISK(110.,75.,10.)",
        [
            MapAxis(np.logspace(0.0, 3.0, 4), name="axis0"),
            MapAxis(np.logspace(0.0, 2.0, 3), name="axis1"),
        ],
    ),
]

hpx_test_geoms = hpx_test_allsky_geoms + hpx_test_partialsky_geoms


def create_map(nside, nested, frame, region, axes):
    return HpxNDMap(
        HpxGeom(nside=nside, nest=nested, frame=frame, region=region, axes=axes)
    )


@pytest.mark.parametrize(("nside", "nested", "frame", "region", "axes"), hpx_test_geoms)
def test_hpxmap_init(nside, nested, frame, region, axes):
    geom = HpxGeom(nside=nside, nest=nested, frame=frame, region=region, axes=axes)
    shape = [int(np.max(geom.npix))]
    if axes:
        shape += [ax.nbin for ax in axes]
    shape = shape[::-1]
    data = np.random.uniform(0, 1, shape)
    m = HpxNDMap(geom)
    assert m.data.shape == data.shape
    m = HpxNDMap(geom, data)
    assert_allclose(m.data, data)


@pytest.mark.parametrize(("nside", "nested", "frame", "region", "axes"), hpx_test_geoms)
def test_hpxmap_create(nside, nested, frame, region, axes):
    create_map(nside, nested, frame, region, axes)


@pytest.mark.parametrize(("nside", "nested", "frame", "region", "axes"), hpx_test_geoms)
def test_hpxmap_read_write(tmp_path, nside, nested, frame, region, axes):
    path = tmp_path / "tmp.fits"

    m = create_map(nside, nested, frame, region, axes)
    m.write(path, sparse=True, overwrite=True)

    m2 = HpxNDMap.read(path)
    m4 = Map.read(path, map_type="hpx")
    msk = np.ones_like(m2.data[...], dtype=bool)

    assert_allclose(m.data[...][msk], m2.data[...][msk])
    assert_allclose(m.data[...][msk], m4.data[...][msk])

    m.write(path, overwrite=True)
    m2 = HpxNDMap.read(path)
    m3 = HpxMap.read(path, map_type="hpx")
    m4 = Map.read(path, map_type="hpx")
    assert_allclose(m.data[...][msk], m2.data[...][msk])
    assert_allclose(m.data[...][msk], m3.data[...][msk])
    assert_allclose(m.data[...][msk], m4.data[...][msk])

    # Specify alternate HDU name for IMAGE and BANDS table
    m.write(path, sparse=True, hdu="IMAGE", hdu_bands="TEST", overwrite=True)
    m2 = HpxNDMap.read(path)
    m3 = Map.read(path)
    m4 = Map.read(path, map_type="hpx")


def test_hpxmap_read_write_fgst(tmp_path):
    path = tmp_path / "tmp.fits"

    axis = MapAxis.from_bounds(100.0, 1000.0, 4, name="energy", unit="MeV")

    # Test Counts Cube
    m = create_map(8, False, "galactic", None, [axis])
    m.write(path, format="fgst-ccube", overwrite=True)
    with fits.open(path, memmap=False) as hdulist:
        assert "SKYMAP" in hdulist
        assert "EBOUNDS" in hdulist
        assert hdulist["SKYMAP"].header["HPX_CONV"] == "FGST-CCUBE"
        assert hdulist["SKYMAP"].header["TTYPE1"] == "CHANNEL1"

    m2 = Map.read(path)
    # TODO: add better asserts here
    assert m2 is not None

    # Test Model Cube
    m.write(path, format="fgst-template", overwrite=True)
    with fits.open(path, memmap=False) as hdulist:
        assert "SKYMAP" in hdulist
        assert "ENERGIES" in hdulist
        assert hdulist["SKYMAP"].header["HPX_CONV"] == "FGST-TEMPLATE"
        assert hdulist["SKYMAP"].header["TTYPE1"] == "ENERGY1"

    m2 = Map.read(path)
    # TODO: add better asserts here
    assert m2 is not None


@requires_data()
def test_read_fgst_exposure():
    exposure = Map.read("$GAMMAPY_DATA/fermi_3fhl/fermi_3fhl_exposure_cube_hpx.fits.gz")
    energy_axis = exposure.geom.axes["energy_true"]
    assert energy_axis.node_type == "center"
    assert exposure.unit == "cm2 s"


@pytest.mark.parametrize(("nside", "nested", "frame", "region", "axes"), hpx_test_geoms)
def test_hpxmap_set_get_by_pix(nside, nested, frame, region, axes):
    m = create_map(nside, nested, frame, region, axes)
    coords = m.geom.get_coord(flat=True)
    idx = m.geom.get_idx(flat=True)
    m.set_by_pix(idx, coords[0])
    assert_allclose(coords[0], m.get_by_pix(idx))


@pytest.mark.parametrize(("nside", "nested", "frame", "region", "axes"), hpx_test_geoms)
def test_hpxmap_set_get_by_coord(nside, nested, frame, region, axes):
    m = create_map(nside, nested, frame, region, axes)
    coords = m.geom.get_coord(flat=True)
    m.set_by_coord(coords, coords[0])
    assert_allclose(coords[0], m.get_by_coord(coords))

    # Test with SkyCoords
    m = create_map(nside, nested, frame, region, axes)
    coords = m.geom.get_coord(flat=True)
    skydir = SkyCoord(coords[0], coords[1], unit="deg", frame=m.geom.frame)
    skydir_cel = skydir.transform_to("icrs")
    skydir_gal = skydir.transform_to("galactic")
    m.set_by_coord((skydir_gal,) + tuple(coords[2:]), coords[0])
    assert_allclose(coords[0], m.get_by_coord(coords))
    assert_allclose(
        m.get_by_coord((skydir_cel,) + tuple(coords[2:])),
        m.get_by_coord((skydir_gal,) + tuple(coords[2:])),
    )


@pytest.mark.parametrize(("nside", "nested", "frame", "region", "axes"), hpx_test_geoms)
def test_hpxmap_interp_by_coord(nside, nested, frame, region, axes):
    m = HpxNDMap(
        HpxGeom(nside=nside, nest=nested, frame=frame, region=region, axes=axes)
    )
    coords = m.geom.get_coord(flat=True)
    m.set_by_coord(coords, coords[1])
    assert_allclose(m.get_by_coord(coords), m.interp_by_coord(coords, method="linear"))


def test_hpxmap_interp_by_coord_quantities():
    ax = MapAxis(np.logspace(0.0, 3.0, 3), interp="log", name="energy", unit="TeV")
    geom = HpxGeom(nside=1, axes=[ax])
    m = HpxNDMap(geom=geom)

    coords_dict = {"lon": 99, "lat": 42, "energy": 1000 * u.GeV}

    coords = m.geom.get_coord(flat=True)
    m.set_by_coord(coords, coords["lat"])

    coords_dict["energy"] = 1 * u.TeV
    val = m.interp_by_coord(coords_dict)
    assert_allclose(val, 42, rtol=1e-2)


@pytest.mark.parametrize(("nside", "nested", "frame", "region", "axes"), hpx_test_geoms)
def test_hpxmap_fill_by_coord(nside, nested, frame, region, axes):
    m = create_map(nside, nested, frame, region, axes)
    coords = m.geom.get_coord(flat=True)
    m.fill_by_coord(coords, coords[1])
    m.fill_by_coord(coords, coords[1])
    assert_allclose(m.get_by_coord(coords), 2.0 * coords[1])


@pytest.mark.parametrize(("nside", "nested", "frame", "region", "axes"), hpx_test_geoms)
def test_hpxmap_to_wcs(nside, nested, frame, region, axes):
    m = HpxNDMap(
        HpxGeom(nside=nside, nest=nested, frame=frame, region=region, axes=axes)
    )
    m.to_wcs(sum_bands=False, oversample=2, normalize=False)
    m.to_wcs(sum_bands=True, oversample=2, normalize=False)


@pytest.mark.parametrize(("nside", "nested", "frame", "region", "axes"), hpx_test_geoms)
def test_hpxmap_swap_scheme(nside, nested, frame, region, axes):
    m = HpxNDMap(
        HpxGeom(nside=nside, nest=nested, frame=frame, region=region, axes=axes)
    )
    m.data = np.arange(m.data.size).reshape(m.geom.data_shape)
    m2 = m.to_swapped()
    coords = m.geom.get_coord(flat=True)
    assert_allclose(m.get_by_coord(coords), m2.get_by_coord(coords))


@pytest.mark.parametrize(
    ("nside", "nested", "frame", "region", "axes"), hpx_test_partialsky_geoms
)
def test_hpxmap_pad(nside, nested, frame, region, axes):
    m = HpxNDMap(
        HpxGeom(nside=nside, nest=nested, frame=frame, region=region, axes=axes)
    )
    m.set_by_pix(m.geom.get_idx(flat=True), 1.0)
    cval = 2.2
    m_pad = m.pad(1, mode="constant", cval=cval)
    coords_pad = m_pad.geom.get_coord(flat=True)
    msk = m.geom.contains(coords_pad)
    coords_out = tuple([c[~msk] for c in coords_pad])
    assert_allclose(m_pad.get_by_coord(coords_out), cval * np.ones_like(coords_out[0]))
    coords_in = tuple([c[msk] for c in coords_pad])
    assert_allclose(m_pad.get_by_coord(coords_in), np.ones_like(coords_in[0]))


def test_hpx_nd_map_pad_axis():
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=2)

    m = HpxNDMap.create(nside=2, frame="galactic", axes=[axis])
    m.data += [[1], [2]]

    m_pad = m.pad(axis_name="energy", pad_width=(1, 1), mode="constant", cval=3)
    assert_allclose(m_pad.data[:, 0], [3, 1, 2, 3])


@pytest.mark.parametrize(
    ("nside", "nested", "frame", "region", "axes"), hpx_test_partialsky_geoms
)
def test_hpxmap_crop(nside, nested, frame, region, axes):
    m = HpxNDMap(
        HpxGeom(nside=nside, nest=nested, frame=frame, region=region, axes=axes)
    )
    m.crop(1)


@pytest.mark.parametrize(("nside", "nested", "frame", "region", "axes"), hpx_test_geoms)
def test_hpxmap_upsample(nside, nested, frame, region, axes):
    m = HpxNDMap(
        HpxGeom(nside=nside, nest=nested, frame=frame, region=region, axes=axes),
        unit="m2",
    )
    m.set_by_pix(m.geom.get_idx(flat=True), 1.0)
    m_up = m.upsample(2, preserve_counts=True)
    assert_allclose(np.nansum(m.data), np.nansum(m_up.data))
    m_up = m.upsample(2, preserve_counts=False)
    assert_allclose(4.0 * np.nansum(m.data), np.nansum(m_up.data))
    assert m.unit == m_up.unit


@pytest.mark.parametrize(("nside", "nested", "frame", "region", "axes"), hpx_test_geoms)
def test_hpxmap_downsample(nside, nested, frame, region, axes):
    m = HpxNDMap(
        HpxGeom(nside=nside, nest=nested, frame=frame, region=region, axes=axes),
        unit="m2",
    )
    m.set_by_pix(m.geom.get_idx(flat=True), 1.0)
    m_down = m.downsample(2, preserve_counts=True)
    assert_allclose(np.nansum(m.data), np.nansum(m_down.data))
    assert m.unit == m_down.unit


@pytest.mark.parametrize(("nside", "nested", "frame", "region", "axes"), hpx_test_geoms)
def test_hpxmap_sum_over_axes(nside, nested, frame, region, axes):
    m = HpxNDMap(
        HpxGeom(nside=nside, nest=nested, frame=frame, region=region, axes=axes)
    )
    coords = m.geom.get_coord(flat=True)
    m.fill_by_coord(coords, coords[0])
    msum = m.sum_over_axes()

    if m.geom.is_regular:
        assert_allclose(np.nansum(m.data), np.nansum(msum.data))


def test_coadd_unit():
    geom = HpxGeom.create(nside=128)
    m1 = HpxNDMap(geom, unit="m2")
    m2 = HpxNDMap(geom, unit="cm2")

    idx = geom.get_idx()

    weights = u.Quantity(np.ones_like(idx[0]), unit="cm2")
    m1.fill_by_idx(idx, weights=weights)
    assert_allclose(m1.data, 0.0001)

    weights = u.Quantity(np.ones_like(idx[0]), unit="m2")
    m1.fill_by_idx(idx, weights=weights)
    m1.coadd(m2)

    assert_allclose(m1.data, 1.0001)


def test_plot():
    m = HpxNDMap.create(binsz=10)
    with mpl_plot_check():
        m.plot()


def test_plot_grid():
    axis = MapAxis([0, 1, 2], node_type="edges")
    m = HpxNDMap.create(binsz=0.1 * u.deg, width=1, axes=[axis])
    with mpl_plot_check():
        m.plot_grid()


def test_plot_poly():
    m = HpxNDMap.create(binsz=10)
    with mpl_plot_check():
        m.plot(method="poly")


def test_hpxndmap_resample_axis():
    axis_1 = MapAxis.from_edges([1, 2, 3, 4, 5], name="test-1")
    axis_2 = MapAxis.from_edges([1, 2, 3, 4], name="test-2")

    geom = HpxGeom.create(nside=16, axes=[axis_1, axis_2])
    m = HpxNDMap(geom, unit="m2")
    m.data += 1

    new_axis = MapAxis.from_edges([2, 3, 5], name="test-1")
    m2 = m.resample_axis(axis=new_axis)
    assert m2.data.shape == (3, 2, 3072)
    assert_allclose(m2.data[0, :, 0], [1, 2])

    # Test without all interval covered
    new_axis = MapAxis.from_edges([1.7, 4], name="test-1")
    m3 = m.resample_axis(axis=new_axis)
    assert m3.data.shape == (3, 1, 3072)
    assert_allclose(m3.data, 2)


def test_hpx_nd_map_to_nside():
    axis = MapAxis.from_edges([1, 2, 3], name="test-1")

    geom = HpxGeom.create(nside=64, axes=[axis])
    m = HpxNDMap(geom, unit="m2")
    m.data += 1

    m2 = m.to_nside(nside=32)
    assert_allclose(m2.data, 4)

    m3 = m.to_nside(nside=128)
    assert_allclose(m3.data, 0.25)


def test_hpx_nd_map_to_wcs_tiles():
    m = HpxNDMap.create(nside=8, frame="galactic")
    m.data += 1

    tiles = m.to_wcs_tiles(nside_tiles=4)
    assert_allclose(tiles[0].data, 1)
    assert_allclose(tiles[32].data, 1)

    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=1)
    m = HpxNDMap.create(nside=8, frame="galactic", axes=[axis])
    m.data += 1

    tiles = m.to_wcs_tiles(nside_tiles=4)
    assert_allclose(tiles[0].data, 1)
    assert_allclose(tiles[32].data, 1)


def test_from_wcs_tiles():
    geom = HpxGeom.create(nside=8)

    wcs_geoms = geom.to_wcs_tiles(nside_tiles=4)

    wcs_tiles = [Map.from_geom(geom, data=1) for geom in wcs_geoms]

    m = HpxNDMap.from_wcs_tiles(wcs_tiles=wcs_tiles)

    assert_allclose(m.data, 1)


def test_hpx_map_cutout():
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=1)
    m = HpxNDMap.create(nside=32, frame="galactic", axes=[axis])
    m.data += np.arange(12288)

    cutout = m.cutout(SkyCoord("0d", "0d"), width=10 * u.deg)

    assert cutout.data.shape == (1, 25)
    assert_allclose(cutout.data.sum(), 239021)
    assert_allclose(cutout.data[0, 0], 8452)
    assert_allclose(cutout.data[0, -1], 9768)


def test_partial_hpx_map_cutout():
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=1)
    m = HpxNDMap.create(
        nside=32, frame="galactic", axes=[axis], region="DISK(110.,75.,10.)"
    )
    m.data += np.arange(90)

    cutout = m.cutout(SkyCoord("0d", "0d"), width=10 * u.deg)

    assert cutout.data.shape == (1, 25)
    assert_allclose(cutout.data.sum(), 2225)
    assert_allclose(cutout.data[0, 0], 89)
    assert_allclose(cutout.data[0, -1], 89)


def test_hpx_map_stack():
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=1)
    m = HpxNDMap.create(
        nside=32, frame="galactic", axes=[axis], region="DISK(110.,75.,10.)"
    )
    m.data += np.arange(90)

    m_allsky = HpxNDMap.create(nside=32, frame="galactic", axes=[axis])
    m_allsky.stack(m)

    assert_allclose(m_allsky.data.sum(), (90 * 89) / 2)

    value = m_allsky.get_by_coord(
        {"skycoord": SkyCoord("110d", "75d", frame="galactic"), "energy": 3 * u.TeV}
    )
    assert_allclose(value, 69)

    with pytest.raises(ValueError):
        m_allsky = HpxNDMap.create(nside=16, frame="galactic", axes=[axis])
        m_allsky.stack(m)


def test_hpx_map_weights_stack():
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=1)
    m = HpxNDMap.create(
        nside=32, frame="galactic", axes=[axis], region="DISK(110.,75.,10.)"
    )
    m.data += np.arange(90) + 1

    weights = m.copy()
    weights.data = 1 / (np.arange(90) + 1)

    m_allsky = HpxNDMap.create(nside=32, frame="galactic", axes=[axis])
    m_allsky.stack(m, weights=weights)

    assert_allclose(m_allsky.data.sum(), 90)


def test_partial_hpx_map_stack():
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=1)
    m_1 = HpxNDMap.create(
        nside=128, frame="galactic", axes=[axis], region="DISK(110.,75.,20.)"
    )
    m_1.data += 1

    m_2 = HpxNDMap.create(
        nside=128, frame="galactic", axes=[axis], region="DISK(130.,75.,20.)"
    )
    m_2.stack(m_1)

    assert_allclose(m_1.data.sum(), 5933)
    assert_allclose(m_2.data.sum(), 4968)


def test_hpx_map_to_region_nd_map():
    axis = MapAxis.from_energy_bounds("10 GeV", "2 TeV", nbin=10)
    m = HpxNDMap.create(nside=128, axes=[axis])
    m.data += 1

    circle = CircleSkyRegion(center=SkyCoord("0d", "0d"), radius=10 * u.deg)

    spec = m.to_region_nd_map(region=circle)
    assert_allclose(spec.data.sum(), 14660)

    spec_mean = m.to_region_nd_map(region=circle, func=np.mean)
    assert_allclose(spec_mean.data, 1)

    spec_interp = m.to_region_nd_map(region=circle.center, func=np.mean)
    assert_allclose(spec_interp.data, 1)


@pytest.mark.parametrize("kernel", ["gauss", "disk"])
def test_smooth(kernel):
    axes = [
        MapAxis(np.logspace(0.0, 3.0, 3), interp="log"),
        MapAxis(np.logspace(1.0, 3.0, 4), interp="lin"),
    ]
    geom_nest = HpxGeom.create(nside=256, nest=False, frame="galactic", axes=axes)
    geom_ring = HpxGeom.create(nside=256, nest=True, frame="galactic", axes=axes)
    m_nest = HpxNDMap(geom_nest, data=np.ones(geom_nest.data_shape), unit="m2")
    m_ring = HpxNDMap(geom_ring, data=np.ones(geom_ring.data_shape), unit="m2")

    desired_nest = m_nest.data.sum()
    desired_ring = m_ring.data.sum()

    smoothed_nest = m_nest.smooth(0.2 * u.deg, kernel)
    smoothed_ring = m_ring.smooth(0.2 * u.deg, kernel)

    actual_nest = smoothed_nest.data.sum()
    assert_allclose(actual_nest, desired_nest)
    assert smoothed_nest.data.dtype == float

    actual_ring = smoothed_ring.data.sum()
    assert_allclose(actual_ring, desired_ring)
    assert smoothed_ring.data.dtype == float

    # with pytest.raises(NotImplementedError):
    cutout = m_nest.cutout(position=(0, 0), width=15 * u.deg)
    smoothed_cutout = cutout.smooth(0.1 * u.deg, kernel)
    actual_cutout = cutout.data.sum()
    desired_cutout = smoothed_cutout.data.sum()
    assert_allclose(actual_cutout, desired_cutout, rtol=0.01)

    with pytest.raises(ValueError):
        m_nest.smooth(0.2 * u.deg, "box")


@pytest.mark.parametrize("nest", [True, False])
def test_convolve_wcs(nest):
    energy = MapAxis.from_bounds(1, 100, unit="TeV", nbin=2, name="energy")
    nside = 256
    hpx_geom = HpxGeom.create(
        nside=nside, axes=[energy], region="DISK(0,0,2.5)", nest=nest
    )
    hpx_map = Map.from_geom(hpx_geom)
    hpx_map.set_by_coord((0, 0, [2, 90]), 1)
    wcs_geom = WcsGeom.create(width=5, binsz=0.04, axes=[energy])

    kernel = PSFKernel.from_gauss(wcs_geom, 0.4 * u.deg)
    convolved_map = hpx_map.convolve_wcs(kernel)
    assert_allclose(convolved_map.data.sum(), 2, rtol=0.001)


@pytest.mark.parametrize("region", [None, "DISK(0,0,70)"])
def test_convolve_full(region):
    energy = MapAxis.from_bounds(1, 100, unit="TeV", nbin=2, name="energy_true")
    nside = 256

    all_sky_geom = HpxGeom(
        nside=nside, axes=[energy], region=region, nest=False, frame="icrs"
    )

    all_sky_map = Map.from_geom(all_sky_geom)
    all_sky_map.set_by_coord((0, 0, [2, 90]), 1)
    all_sky_map.set_by_coord((10, 10, [2, 90]), 1)
    all_sky_map.set_by_coord((30, 30, [2, 90]), 1)
    all_sky_map.set_by_coord((-40, -40, [2, 90]), 1)
    all_sky_map.set_by_coord((60, 0, [2, 90]), 1)
    all_sky_map.set_by_coord((-45, 30, [2, 90]), 1)
    all_sky_map.set_by_coord((30, -45, [2, 90]), 1)

    wcs_geom = WcsGeom.create(width=5, binsz=0.05, axes=[energy])
    psf = PSFMap.from_gauss(energy_axis_true=energy, sigma=[0.5, 0.6] * u.deg)

    kernel = psf.get_psf_kernel(geom=wcs_geom, max_radius=1 * u.deg)
    convolved_map = all_sky_map.convolve_full(kernel)
    assert_allclose(convolved_map.data.sum(), 14, rtol=1e-5)


def test_hpxmap_read_healpy(tmp_path):
    import healpy as hp

    path = tmp_path / "tmp.fits"
    npix = 12 * 1024 * 1024
    m = [np.arange(npix), np.arange(npix) - 1, np.arange(npix) - 2]
    hp.write_map(filename=path, m=m, nest=False, overwrite=True)
    with fits.open(path, memmap=False) as hdulist:
        hdu_out = find_bintable_hdu(hdulist)
        header = hdu_out.header
        assert header["PIXTYPE"] == "HEALPIX"
        assert header["ORDERING"] == "RING"
        assert header["EXTNAME"] == "xtension"
        assert header["NSIDE"] == 1024
        format = HpxConv.identify_hpx_format(header)
        assert format == "healpy"

    # first column "TEMPERATURE"
    m1 = Map.read(path, colname="TEMPERATURE")
    assert m1.data.shape[0] == npix
    diff = np.sum(m[0] - m1.data)
    assert_allclose(diff, 0.0)

    # specifying the colname by default for healpy it is "Q_POLARISATION"
    m2 = Map.read(path, colname="Q_POLARISATION")
    assert m2.data.shape[0] == npix
    diff = np.sum(m[1] - m2.data)
    assert_allclose(diff, 0.0)


def test_map_plot_mask():
    geom = HpxGeom.create(nside=16)

    region = CircleSkyRegion(
        center=SkyCoord("0d", "0d", frame="galactic"), radius=20 * u.deg
    )

    mask = geom.region_mask([region])

    with mpl_plot_check():
        mask.plot_mask()


def test_hpx_map_sampling():
    hpxmap = HpxNDMap.create(nside=16)
    with pytest.raises(NotImplementedError):
        hpxmap.sample_coord(2)

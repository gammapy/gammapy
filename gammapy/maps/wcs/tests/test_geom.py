# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from regions import CircleSkyRegion
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.maps.utils import _check_binsz, _check_width

axes1 = [MapAxis(np.logspace(0.0, 3.0, 3), interp="log", name="energy")]
axes2 = [
    MapAxis(np.logspace(0.0, 3.0, 3), interp="log", name="energy"),
    MapAxis(np.logspace(1.0, 3.0, 4), interp="lin"),
]
skydir = SkyCoord(110.0, 75.0, unit="deg", frame="icrs")

wcs_allsky_test_geoms = [
    (None, 10.0, "galactic", "AIT", skydir, None),
    (None, 10.0, "galactic", "AIT", skydir, axes1),
    (None, [10.0, 20.0], "galactic", "AIT", skydir, axes1),
    (None, 10.0, "galactic", "AIT", skydir, axes2),
    (None, [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]], "galactic", "AIT", skydir, axes2),
]

wcs_partialsky_test_geoms = [
    (10, 0.1, "galactic", "AIT", skydir, None),
    (10, 0.1, "galactic", "AIT", skydir, axes1),
    (10, [0.1, 0.2], "galactic", "AIT", skydir, axes1),
]

wcs_test_geoms = wcs_allsky_test_geoms + wcs_partialsky_test_geoms


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsgeom_init(npix, binsz, frame, proj, skydir, axes):
    WcsGeom.create(
        npix=npix, binsz=binsz, skydir=skydir, proj=proj, frame=frame, axes=axes
    )


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsgeom_get_pix(npix, binsz, frame, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, skydir=skydir, proj=proj, frame=frame, axes=axes
    )
    pix = geom.get_idx()
    if axes is not None:
        idx = tuple([1] * len(axes))
        pix_img = geom.get_idx(idx=idx)
        m = np.all(np.stack([x == y for x, y in zip(idx, pix[2:])]), axis=0)
        m2 = pix_img[0] != -1
        assert_allclose(pix[0][m], np.ravel(pix_img[0][m2]))
        assert_allclose(pix[1][m], np.ravel(pix_img[1][m2]))


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsgeom_test_pix_to_coord(npix, binsz, frame, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, skydir=skydir, proj=proj, frame=frame, axes=axes
    )
    assert_allclose(geom.get_coord()[0], geom.pix_to_coord(geom.get_idx())[0])


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsgeom_test_coord_to_idx(npix, binsz, frame, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz, proj=proj, frame=frame, axes=axes)
    assert_allclose(geom.get_idx()[0], geom.coord_to_idx(geom.get_coord())[0])

    if not geom.is_allsky:
        coords = geom.center_coord[:2] + tuple([ax.center[0] for ax in geom.axes])
        coords[0][...] += 2.0 * np.max(geom.width[0])
        idx = geom.coord_to_idx(coords)
        assert_allclose(np.full_like(coords[0].value, -1, dtype=int), idx[0])
        idx = geom.coord_to_idx(coords, clip=True)
        assert np.all(
            np.not_equal(np.full_like(coords[0].value, -1, dtype=int), idx[0])
        )


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsgeom_read_write(tmp_path, npix, binsz, frame, proj, skydir, axes):
    geom0 = WcsGeom.create(npix=npix, binsz=binsz, proj=proj, frame=frame, axes=axes)

    hdu_bands = geom0.to_bands_hdu(hdu_bands="TEST_BANDS")
    hdu_prim = fits.PrimaryHDU()
    hdu_prim.header.update(geom0.to_header())

    hdulist = fits.HDUList([hdu_prim, hdu_bands])
    hdulist.writeto(tmp_path / "tmp.fits")

    with fits.open(tmp_path / "tmp.fits", memmap=False) as hdulist:
        geom1 = WcsGeom.from_header(hdulist[0].header, hdulist["TEST_BANDS"])

    assert_allclose(geom0.npix, geom1.npix)
    assert geom0.frame == geom1.frame


def test_wcsgeom_to_hdulist():
    npix, binsz, frame, proj, skydir, axes = wcs_test_geoms[3]
    geom = WcsGeom.create(npix=npix, binsz=binsz, proj=proj, frame=frame, axes=axes)

    hdu = geom.to_bands_hdu(hdu_bands="TEST")
    assert hdu.header["AXCOLS1"] == "E_MIN,E_MAX"
    assert hdu.header["AXCOLS2"] == "AXIS1_MIN,AXIS1_MAX"


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsgeom_contains(npix, binsz, frame, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, skydir=skydir, proj=proj, frame=frame, axes=axes
    )
    coords = geom.get_coord()
    m = np.isfinite(coords[0])
    coords = [c[m] for c in coords]
    assert_allclose(geom.contains(coords), np.ones(coords[0].shape, dtype=bool))

    if axes is not None:
        coords = [c[0] for c in coords[:2]] + [ax.edges[-1] + 1.0 for ax in axes]
        assert_allclose(geom.contains(coords), np.zeros((1,), dtype=bool))

    if not geom.is_allsky:
        coords = [0.0, 0.0] + [ax.center[0] for ax in geom.axes]
        assert_allclose(geom.contains(coords), np.zeros((1,), dtype=bool))


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcs_geom_from_aligned(npix, binsz, frame, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, skydir=(0, 0), proj=proj, frame=frame, axes=axes
    )

    aligned_geom = WcsGeom.from_aligned(geom=geom, skydir=(2, 3), width="90 deg")

    assert aligned_geom.is_aligned(geom)


def test_from_aligned_vs_cutout():
    skydir = SkyCoord(0.12, -0.34, unit="deg", frame="galactic")

    geom = WcsGeom.create(binsz=0.1, skydir=skydir, proj="AIT", frame="galactic")

    position = SkyCoord("2.23d", "3.102d", frame="galactic")

    width = ("89 deg", "79 deg")
    aligned_geom = WcsGeom.from_aligned(geom=geom, skydir=position, width=width)

    geom_cutout = geom.cutout(position=position, width=width)

    assert geom_cutout == aligned_geom


def test_from_aligned_vs_cutout_tan():
    skydir = SkyCoord(0, 0, unit="deg", frame="galactic")

    geom = WcsGeom.create(
        binsz=1, skydir=skydir, proj="TAN", frame="galactic", width=("180d", "90d")
    )

    position = SkyCoord("53.23d", "22.102d", frame="galactic")
    width = ("17 deg", "15 deg")

    geom_cutout = geom.cutout(position=position, width=width, mode="partial")

    aligned_geom = WcsGeom.from_aligned(geom=geom, skydir=position, width=width)

    assert aligned_geom == geom_cutout


def test_wcsgeom_solid_angle():
    # Test using a CAR projection map with an extra axis
    binsz = 1.0 * u.deg
    npix = 10
    geom = WcsGeom.create(
        skydir=(0, 0),
        npix=(npix, npix),
        binsz=binsz,
        frame="galactic",
        proj="CAR",
        axes=[MapAxis.from_edges([0, 2, 3])],
    )

    solid_angle = geom.solid_angle()

    # Check array size
    assert solid_angle.shape == (2, npix, npix)

    # Test at b = 0 deg
    assert solid_angle.unit == "sr"
    assert_allclose(solid_angle.value[0, 5, 5], 0.0003046, rtol=1e-3)

    # Test at b = 5 deg
    assert_allclose(solid_angle.value[0, 9, 5], 0.0003038, rtol=1e-3)


def test_wcsgeom_solid_angle_symmetry():
    geom = WcsGeom.create(
        skydir=(0, 0), frame="galactic", npix=(3, 3), binsz=20.0 * u.deg
    )

    sa = geom.solid_angle()

    assert_allclose(sa[1, :], sa[1, 0])  # Constant along lon
    assert_allclose(sa[0, 1], sa[2, 1])  # Symmetric along lat
    with pytest.raises(AssertionError):
        # Not constant along lat due to changes in solid angle (great circle)
        assert_allclose(sa[:, 1], sa[0, 1])


def test_wcsgeom_solid_angle_ait():
    # Pixels that don't correspond to locations on the sky
    # should have solid angles set to NaN
    ait_geom = WcsGeom.create(
        skydir=(0, 0), width=(360, 180), binsz=20, frame="galactic", proj="AIT"
    )
    solid_angle = ait_geom.solid_angle().to_value("deg2")

    assert_allclose(solid_angle[4, 1], 397.04838)
    assert_allclose(solid_angle[4, 16], 397.751841)
    assert_allclose(solid_angle[1, 8], 381.556269)
    assert_allclose(solid_angle[7, 8], 398.34725)

    assert np.isnan(solid_angle[0, 0])


def test_wcsgeom_separation():
    geom = WcsGeom.create(
        skydir=(0, 0),
        npix=10,
        binsz=0.1,
        frame="galactic",
        proj="CAR",
        axes=[MapAxis.from_edges([0, 2, 3])],
    )
    position = SkyCoord(1, 0, unit="deg", frame="galactic").icrs
    separation = geom.separation(position)

    assert separation.unit == "deg"
    assert separation.shape == (10, 10)
    assert_allclose(separation.value[0, 0], 0.7106291438079875)

    # Make sure it also works for 2D maps as input
    separation = geom.to_image().separation(position)
    assert separation.unit == "deg"
    assert separation.shape == (10, 10)
    assert_allclose(separation.value[0, 0], 0.7106291438079875)


def test_cutout():
    geom = WcsGeom.create(
        skydir=(0, 0),
        npix=10,
        binsz=0.1,
        frame="galactic",
        proj="CAR",
        axes=[MapAxis.from_edges([0, 2, 3])],
    )
    position = SkyCoord(0.1, 0.2, unit="deg", frame="galactic")
    cutout_geom = geom.cutout(position=position, width=2 * 0.3 * u.deg, mode="trim")

    center_coord = cutout_geom.center_coord
    assert_allclose(center_coord[0].value, 0.1)
    assert_allclose(center_coord[1].value, 0.2)
    assert_allclose(center_coord[2].value, 2.0)

    assert cutout_geom.data_shape == (2, 6, 6)
    assert cutout_geom.data_shape_axes == (2, 1, 1)


def test_cutout_info():
    geom = WcsGeom.create(skydir=(0, 0), npix=10)
    position = SkyCoord(0, 0, unit="deg")
    cutout_geom = geom.cutout(position=position, width="2 deg")

    cutout_info = cutout_geom.cutout_slices(geom)

    assert cutout_info["parent-slices"][0].start == 3
    assert cutout_info["parent-slices"][1].start == 3

    assert cutout_info["cutout-slices"][0].start == 0
    assert cutout_info["cutout-slices"][1].start == 0


def test_cutout_min_size():
    geom = WcsGeom.create(skydir=(0, 0), npix=10, binsz=0.5)
    position = SkyCoord(0, 0, unit="deg")
    cutout_geom = geom.cutout(position=position, width=["2 deg", "0.1 deg"])

    assert cutout_geom.data_shape == (1, 4)


def test_wcs_geom_get_coord():
    geom = WcsGeom.create(
        skydir=(0, 0), npix=(4, 3), binsz=1, frame="galactic", proj="CAR"
    )
    coord = geom.get_coord(mode="edges")
    assert_allclose(coord.lon[0, 0].value, 2)
    assert coord.lon[0, 0].unit == "deg"
    assert_allclose(coord.lat[0, 0].value, -1.5)
    assert coord.lat[0, 0].unit == "deg"


def test_wcs_geom_instance_cache():
    geom_1 = WcsGeom.create(npix=(3, 3))
    geom_2 = WcsGeom.create(npix=(3, 3))

    coord_1, coord_2 = geom_1.get_coord(), geom_2.get_coord()

    assert geom_1.get_coord.cache_info().misses == 1
    assert geom_2.get_coord.cache_info().misses == 1

    coord_1_cached, coord_2_cached = geom_1.get_coord(), geom_2.get_coord()

    assert geom_1.get_coord.cache_info().hits == 1
    assert geom_2.get_coord.cache_info().hits == 1

    assert geom_1.get_coord.cache_info().currsize == 1
    assert geom_2.get_coord.cache_info().currsize == 1

    assert id(coord_1) == id(coord_1_cached)
    assert id(coord_2) == id(coord_2_cached)


def test_wcs_geom_squash():
    axis = MapAxis.from_nodes([1, 2, 3], name="test-axis")
    geom = WcsGeom.create(npix=(3, 3), axes=[axis])
    geom_squashed = geom.squash(axis_name="test-axis")
    assert geom_squashed.data_shape == (1, 3, 3)


def test_wcs_geom_drop():
    ax1 = MapAxis.from_nodes([1, 2, 3], name="ax1")
    ax2 = MapAxis.from_nodes([1, 2], name="ax2")
    ax3 = MapAxis.from_nodes([1, 2, 3, 4], name="ax3")
    geom = WcsGeom.create(npix=(3, 3), axes=[ax1, ax2, ax3])
    geom_drop = geom.drop(axis_name="ax1")
    assert geom_drop.data_shape == (4, 2, 3, 3)


def test_wcs_geom_resample_overflows():
    ax1 = MapAxis.from_edges([1, 2, 3, 4, 5], name="ax1")
    ax2 = MapAxis.from_nodes([1, 2, 3], name="ax2")
    geom = WcsGeom.create(npix=(3, 3), axes=[ax1, ax2])
    new_axis = MapAxis.from_edges([-1.0, 1, 2.3, 4.8, 6], name="ax1")
    geom_resample = geom.resample_axis(axis=new_axis)

    assert geom_resample.data_shape == (3, 2, 3, 3)
    assert geom_resample.data_shape_axes == (3, 2, 1, 1)
    assert_allclose(geom_resample.axes[0].edges, [1, 2, 5])


def test_wcs_geom_get_pix_coords():
    geom = WcsGeom.create(
        skydir=(0, 0), npix=(4, 3), binsz=1, frame="galactic", proj="CAR", axes=axes1
    )
    idx_center = geom.get_pix(mode="center")

    for idx in idx_center:
        assert idx.shape == (2, 3, 4)
        assert_allclose(idx[0, 0, 0], 0)

    idx_edge = geom.get_pix(mode="edges")
    for idx, desired in zip(idx_edge, [-0.5, -0.5, 0]):
        assert idx.shape == (2, 4, 5)
        assert_allclose(idx[0, 0, 0], desired)


def test_geom_repr():
    geom = WcsGeom.create(
        skydir=(0, 0), npix=(10, 4), binsz=50, frame="galactic", proj="AIT"
    )

    str_info = repr(geom)
    assert geom.__class__.__name__ in str_info
    assert "wcs ref" in str_info
    assert "center" in str_info
    assert "projection" in str_info
    assert "axes" in str_info
    assert "shape" in str_info
    assert "ndim" in str_info
    assert "width" in str_info


def test_geom_refpix():
    refpix = (400, 300)
    geom = WcsGeom.create(
        skydir=(0, 0), npix=(800, 600), refpix=refpix, binsz=0.1, frame="galactic"
    )
    assert_allclose(geom.wcs.wcs.crpix, refpix)


def test_region_mask():
    geom = WcsGeom.create(npix=(3, 3), binsz=2, proj="CAR")

    r1 = CircleSkyRegion(SkyCoord(0, 0, unit="deg"), 1 * u.deg)
    r2 = CircleSkyRegion(SkyCoord(20, 20, unit="deg"), 1 * u.deg)
    regions = [r1, r2]

    mask = geom.region_mask(regions)
    assert mask.data.dtype == bool
    assert np.sum(mask.data) == 1

    mask = geom.region_mask(regions, inside=False)
    assert np.sum(mask.data) == 8


def test_energy_mask():
    energy_axis = MapAxis.from_nodes(
        [1, 10, 100], interp="log", name="energy", unit="TeV"
    )
    geom = WcsGeom.create(npix=(1, 1), binsz=1, proj="CAR", axes=[energy_axis])

    mask = geom.energy_mask(energy_min=3 * u.TeV).data
    assert not mask[0, 0, 0]
    assert mask[1, 0, 0]
    assert mask[2, 0, 0]

    mask = geom.energy_mask(energy_max=30 * u.TeV).data
    assert mask[0, 0, 0]
    assert not mask[1, 0, 0]
    assert not mask[2, 0, 0]

    mask = geom.energy_mask(energy_min=3 * u.TeV, energy_max=40 * u.TeV).data
    assert not mask[0, 0, 0]
    assert not mask[2, 0, 0]
    assert mask[1, 0, 0]


def test_boundary_mask():
    axis = MapAxis.from_edges([1, 10, 100])
    geom = WcsGeom.create(
        skydir=(0, 0),
        binsz=0.02,
        width=(2, 2),
        axes=[axis],
    )

    mask = geom.boundary_mask(width=(0.3 * u.deg, 0.1 * u.deg))
    assert np.sum(mask.data[0, :, :]) == 6300
    assert np.sum(mask.data[1, :, :]) == 6300


@pytest.mark.parametrize(
    ("width", "out"),
    [
        (10, (10, 10)),
        ((10 * u.deg).to("rad"), (10, 10)),
        ((10, 5), (10, 5)),
        (("10 deg", "5 deg"), (10, 5)),
        (Angle([10, 5], "deg"), (10, 5)),
        ((10 * u.deg, 5 * u.deg), (10, 5)),
        ((10, 5) * u.deg, (10, 5)),
        ([10, 5], (10, 5)),
        (["10 deg", "5 deg"], (10, 5)),
        (np.array([10, 5]), (10, 5)),
    ],
)
def test_check_width(width, out):
    width = _check_width(width)
    assert isinstance(width, tuple)
    assert isinstance(width[0], float)
    assert isinstance(width[1], float)
    assert width == out

    geom = WcsGeom.create(width=width, binsz=1.0)
    assert tuple(geom.npix) == out


def test_check_binsz():
    # float
    binsz = _check_binsz(0.1)
    assert isinstance(binsz, float)
    # string and other units
    binsz = _check_binsz("0.1deg")
    assert isinstance(binsz, float)
    binsz = _check_binsz("3.141592653589793 rad")
    assert_allclose(binsz, 180)
    # tuple
    binsz = _check_binsz(("0.1deg", "0.2deg"))
    assert isinstance(binsz, tuple)
    assert isinstance(binsz[0], float)
    assert isinstance(binsz[1], float)
    # list
    binsz = _check_binsz(["0.1deg", "0.2deg"])
    assert isinstance(binsz, list)
    assert isinstance(binsz[0], float)
    assert isinstance(binsz[1], float)


def test_check_width_bad_input():
    with pytest.raises(IndexError):
        _check_width(width=(10,))


def test_get_axis_index_by_name():
    e_axis = MapAxis.from_edges([1, 5], name="energy")
    geom = WcsGeom.create(width=5, binsz=1.0, axes=[e_axis])
    assert geom.axes.index("energy") == 0
    with pytest.raises(ValueError):
        geom.axes.index("time")


test_axis1 = [MapAxis(nodes=(1, 2, 3, 4), unit="TeV", node_type="center")]
test_axis2 = [
    MapAxis(nodes=(1, 2, 3, 4), unit="TeV", node_type="center"),
    MapAxis(nodes=(1, 2, 3), unit="TeV", node_type="center"),
]

skydir2 = SkyCoord(110.0, 75.0 + 1e-8, unit="deg", frame="icrs")
skydir3 = SkyCoord(110.0, 75.0 + 1e-3, unit="deg", frame="icrs")

compatibility_test_geoms = [
    (10, 0.1, "galactic", "CAR", skydir, test_axis1, True),
    (10, 0.1, "galactic", "CAR", skydir2, test_axis1, True),
    (10, 0.1, "galactic", "CAR", skydir3, test_axis1, False),
    (10, 0.1, "galactic", "TAN", skydir, test_axis1, False),
    (8, 0.1, "galactic", "CAR", skydir, test_axis1, False),
    (10, 0.1, "galactic", "CAR", skydir, test_axis2, False),
    (10, 0.1, "galactic", "CAR", skydir.galactic, test_axis1, True),
]


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skypos", "axes", "result"),
    compatibility_test_geoms,
)
def test_wcs_geom_equal(npix, binsz, frame, proj, skypos, axes, result):
    geom0 = WcsGeom.create(
        skydir=skydir, npix=10, binsz=0.1, proj="CAR", frame="galactic", axes=test_axis1
    )
    geom1 = WcsGeom.create(
        skydir=skypos, npix=npix, binsz=binsz, proj=proj, frame=frame, axes=axes
    )

    assert (geom0 == geom1) is result
    assert (geom0 != geom1) is not result


def test_irregular_geom_equality():
    axis = MapAxis.from_bounds(1, 3, 10, name="axis", unit="")
    geom0 = WcsGeom.create(skydir=(0, 0), npix=10, binsz=0.1, axes=[axis])
    binsizes = np.ones((10)) * 0.1
    geom1 = WcsGeom.create(skydir=(0, 0), npix=10, binsz=binsizes, axes=[axis])

    with pytest.raises(NotImplementedError):
        geom0 == geom1


def test_wcs_geom_pad():
    axis = MapAxis.from_bounds(0, 1, nbin=1, name="axis", unit="")
    geom = WcsGeom.create(skydir=(0, 0), npix=10, binsz=0.1, axes=[axis])

    geom_pad = geom.pad(axis_name="axis", pad_width=1)
    assert_allclose(geom_pad.axes["axis"].edges, [-1, 0, 1, 2])


@pytest.mark.parametrize("node_type", ["edges", "center"])
@pytest.mark.parametrize("interp", ["log", "lin", "sqrt"])
def test_read_write(tmp_path, node_type, interp):
    # Regression test for MapAxis interp and node_type FITS serialization
    # https://github.com/gammapy/gammapy/issues/1887
    e_ax = MapAxis([1, 2], interp, "energy", node_type, "TeV")
    t_ax = MapAxis([3, 4], interp, "time", node_type, "s")
    m = Map.create(binsz=1, npix=10, axes=[e_ax, t_ax], unit="m2")

    # Check what Gammapy writes in the FITS header
    header = m.to_hdu().header
    assert header["INTERP1"] == interp
    assert header["INTERP2"] == interp

    # Check that all MapAxis properties are preserved on FITS I/O
    m.write(tmp_path / "tmp.fits", overwrite=True)
    m2 = Map.read(tmp_path / "tmp.fits")
    assert m2.geom == m.geom


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skypos", "axes", "result"),
    compatibility_test_geoms,
)
def test_wcs_geom_to_binsz(npix, binsz, frame, proj, skypos, axes, result):
    geom = WcsGeom.create(
        skydir=skydir, npix=10, binsz=0.1, proj="CAR", frame="galactic", axes=test_axis1
    )

    geom_new = geom.to_binsz(binsz=0.5)

    assert_allclose(geom_new.pixel_scales.value, 0.5)


def test_non_equal_binsz():
    geom = WcsGeom.create(
        width=(360, 180), binsz=(360, 60), frame="icrs", skydir=(0, 0), proj="CAR"
    )

    coords = geom.get_coord()

    assert_allclose(coords["lon"].to_value("deg"), 0)
    assert_allclose(coords["lat"].to_value("deg"), [[-60], [0], [60]])


def test_wcs_geom_to_even_npix():
    geom = WcsGeom.create(skydir=(0, 0), binsz=1, width=(3, 3))

    geom_even = geom.to_even_npix()

    assert geom_even.data_shape == (4, 4)

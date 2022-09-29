# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
from astropy.convolution import Box2DKernel, Gaussian2DKernel
from astropy.coordinates import SkyCoord
from astropy.io import fits
from regions import CircleSkyRegion, PointSkyRegion, RectangleSkyRegion
from gammapy.datasets.map import MapEvaluator
from gammapy.irf import PSFKernel, PSFMap
from gammapy.maps import Map, MapAxis, MapCoord, WcsGeom, WcsNDMap
from gammapy.modeling.models import (
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.testing import mpl_plot_check, requires_data

axes1 = [MapAxis(np.logspace(0.0, 3.0, 3), interp="log", name="spam")]
axes2 = [
    MapAxis(np.logspace(0.0, 3.0, 3), interp="log"),
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
    (10, 1.0, "galactic", "AIT", skydir, None),
    (10, 1.0, "galactic", "AIT", skydir, axes1),
    (10, [1.0, 2.0], "galactic", "AIT", skydir, axes1),
    (10, 1.0, "galactic", "AIT", skydir, axes2),
    (10, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], "galactic", "AIT", skydir, axes2),
]

wcs_test_geoms = wcs_allsky_test_geoms + wcs_partialsky_test_geoms


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_init(npix, binsz, frame, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz, proj=proj, frame=frame, axes=axes)
    m0 = WcsNDMap(geom)
    coords = m0.geom.get_coord()
    m0.set_by_coord(coords, coords[1])
    m1 = WcsNDMap(geom, m0.data)
    assert_allclose(m0.data, m1.data)


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_read_write(tmp_path, npix, binsz, frame, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz, proj=proj, frame=frame, axes=axes)
    path = tmp_path / "tmp.fits"

    m0 = WcsNDMap(geom)
    m0.write(path, overwrite=True)
    m1 = WcsNDMap.read(path)
    m2 = Map.read(path)
    m3 = Map.read(path, map_type="wcs")
    assert_allclose(m0.data, m1.data)
    assert_allclose(m0.data, m2.data)
    assert_allclose(m0.data, m3.data)

    m0.write(path, sparse=True, overwrite=True)
    m1 = WcsNDMap.read(path)
    m2 = Map.read(path)
    m3 = Map.read(path, map_type="wcs")
    assert_allclose(m0.data, m1.data)
    assert_allclose(m0.data, m2.data)
    assert_allclose(m0.data, m3.data)

    # Specify alternate HDU name for IMAGE and BANDS table
    m0.write(path, hdu="IMAGE", hdu_bands="TEST", overwrite=True)
    m1 = WcsNDMap.read(path)
    m2 = Map.read(path)
    m3 = Map.read(path, map_type="wcs")


def test_wcsndmap_read_write_fgst(tmp_path):
    path = tmp_path / "tmp.fits"

    axis = MapAxis.from_bounds(100.0, 1000.0, 4, name="energy", unit="MeV")
    geom = WcsGeom.create(npix=10, binsz=1.0, proj="AIT", frame="galactic", axes=[axis])

    # Test Counts Cube
    m = WcsNDMap(geom)
    m.write(path, format="fgst-ccube", overwrite=True)
    with fits.open(path, memmap=False) as hdulist:
        assert "EBOUNDS" in hdulist

    m2 = Map.read(path)
    assert m2.geom.axes[0].name == "energy"

    # Test Model Cube
    m.write(path, format="fgst-template", overwrite=True)
    with fits.open(path, memmap=False) as hdulist:
        assert "ENERGIES" in hdulist


@requires_data()
def test_wcsndmap_read_ccube():
    counts = Map.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-counts-cube.fits.gz")
    energy_axis = counts.geom.axes["energy"]
    # for the 3FGL data the lower energy threshold should be at 10 GeV
    assert_allclose(energy_axis.edges.min().to_value("GeV"), 10, rtol=1e-3)


@requires_data()
def test_wcsndmap_read_exposure():
    exposure = Map.read(
        "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-exposure-cube.fits.gz"
    )
    energy_axis = exposure.geom.axes["energy_true"]
    assert energy_axis.node_type == "center"
    assert exposure.unit == "cm2 s"


def test_wcs_nd_map_data_transpose_issue(tmp_path):
    # Regression test for https://github.com/gammapy/gammapy/issues/1346

    # Our test case: a little map with WCS shape (3, 2), i.e. numpy array shape (2, 3)
    data = np.array([[0, 1, 2], [np.nan, np.inf, -np.inf]])
    geom = WcsGeom.create(npix=(3, 2))

    # Data should be unmodified after init
    m = WcsNDMap(data=data, geom=geom)
    assert_equal(m.data, data)

    # Data should be unmodified if initialised like this
    m = WcsNDMap(geom=geom)
    # and then filled via an in-place Numpy array operation
    m.data += data
    assert_equal(m.data, data)

    # Data should be unmodified after write / read to normal image format
    m.write(tmp_path / "normal.fits.gz")
    m2 = Map.read(tmp_path / "normal.fits.gz")
    assert_equal(m2.data, data)

    # Data should be unmodified after write / read to sparse image format
    m.write(tmp_path / "sparse.fits.gz", sparse=True)
    m2 = Map.read(tmp_path / "sparse.fits.gz")
    assert_equal(m2.data, data)


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_set_get_by_pix(npix, binsz, frame, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, skydir=skydir, proj=proj, frame=frame, axes=axes
    )
    m = WcsNDMap(geom)
    coords = m.geom.get_coord()
    pix = m.geom.get_idx()
    m.set_by_pix(pix, coords[0])
    assert_allclose(coords[0].value, m.get_by_pix(pix))


def test_get_by_coord_bool_int():
    mask = WcsNDMap.create(width=2, dtype="bool")
    coords = {"lon": [0, 3], "lat": [0, 3]}
    vals = mask.get_by_coord(coords)
    assert_allclose(vals, [0, np.nan])

    mask = WcsNDMap.create(width=2, dtype="int")
    coords = {"lon": [0, 3], "lat": [0, 3]}
    vals = mask.get_by_coord(coords)
    assert_allclose(vals, [0, np.nan])


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_set_get_by_coord(npix, binsz, frame, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, skydir=skydir, proj=proj, frame=frame, axes=axes
    )
    m = WcsNDMap(geom)
    coords = m.geom.get_coord()
    m.set_by_coord(coords, coords[0])
    assert_allclose(coords[0].value, m.get_by_coord(coords))

    # Test with SkyCoords
    m = WcsNDMap(geom)
    coords = m.geom.get_coord()
    skydir = coords.skycoord
    skydir_cel = skydir.transform_to("icrs")
    skydir_gal = skydir.transform_to("galactic")

    m.set_by_coord((skydir_gal,) + tuple(coords[2:]), coords[0])
    assert_allclose(coords[0].value, m.get_by_coord(coords))
    assert_allclose(
        m.get_by_coord((skydir_cel,) + tuple(coords[2:])),
        m.get_by_coord((skydir_gal,) + tuple(coords[2:])),
    )

    # Test with MapCoord
    m = WcsNDMap(geom)
    coords = m.geom.get_coord()
    coords_dict = dict(lon=coords[0], lat=coords[1])
    if axes:
        for i, ax in enumerate(axes):
            coords_dict[ax.name] = coords[i + 2]
    map_coords = MapCoord.create(coords_dict, frame=frame)
    m.set_by_coord(map_coords, coords[0])
    assert_allclose(coords[0].value, m.get_by_coord(map_coords))


def test_set_get_by_coord_quantities():
    ax = MapAxis(np.logspace(0.0, 3.0, 3), interp="log", name="energy", unit="TeV")
    geom = WcsGeom.create(binsz=0.1, npix=(3, 4), axes=[ax])
    m = WcsNDMap(geom)
    coords_dict = {"lon": 0, "lat": 0, "energy": 1000 * u.GeV}

    m.set_by_coord(coords_dict, 42)

    coords_dict["energy"] = 1 * u.TeV
    assert_allclose(42, m.get_by_coord(coords_dict))


def qconcatenate(q_1, q_2):
    """Concatenate quantity"""
    return u.Quantity(np.concatenate((q_1.value, q_2.value)), unit=q_1.unit)


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_fill_by_coord(npix, binsz, frame, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, skydir=skydir, proj=proj, frame=frame, axes=axes
    )
    m = WcsNDMap(geom)
    coords = m.geom.get_coord()
    fill_coords = tuple([qconcatenate(t, t) for t in coords])

    fill_vals = fill_coords[1]
    m.fill_by_coord(fill_coords, fill_vals.value)
    assert_allclose(m.get_by_coord(coords), 2.0 * coords[1].value)

    # Test with SkyCoords
    m = WcsNDMap(geom)
    coords = m.geom.get_coord()
    skydir = coords.skycoord
    skydir_cel = skydir.transform_to("icrs")
    skydir_gal = skydir.transform_to("galactic")
    fill_coords_cel = (skydir_cel,) + tuple(coords[2:])
    fill_coords_gal = (skydir_gal,) + tuple(coords[2:])
    m.fill_by_coord(fill_coords_cel, coords[1].value)
    m.fill_by_coord(fill_coords_gal, coords[1].value)
    assert_allclose(m.get_by_coord(coords), 2.0 * coords[1].value)


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_coadd(npix, binsz, frame, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, skydir=skydir, proj=proj, frame=frame, axes=axes
    )
    m0 = WcsNDMap(geom)
    m1 = WcsNDMap(geom.upsample(2))
    coords = m0.geom.get_coord()
    m1.fill_by_coord(
        tuple([qconcatenate(t, t) for t in coords]),
        qconcatenate(coords[1], coords[1]).value,
    )
    m0.coadd(m1)
    assert_allclose(np.nansum(m0.data), np.nansum(m1.data), rtol=1e-4)


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_interp_by_coord(npix, binsz, frame, proj, skydir, axes):
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, skydir=skydir, proj=proj, frame=frame, axes=axes
    )
    m = WcsNDMap(geom)
    coords = m.geom.get_coord().flat
    m.set_by_coord(coords, coords[1].value)
    assert_allclose(coords[1].value, m.interp_by_coord(coords, method="nearest"))


def test_interp_by_coord_quantities():
    ax = MapAxis(
        np.logspace(0.0, 3.0, 3),
        interp="log",
        name="energy",
        unit="TeV",
        node_type="center",
    )
    geom = WcsGeom.create(binsz=0.1, npix=(3, 3), axes=[ax])
    m = WcsNDMap(geom)
    coords_dict = {"lon": 0, "lat": 0, "energy": 1000 * u.GeV}

    m.set_by_coord(coords_dict, 42)

    coords_dict["energy"] = 1 * u.TeV
    assert_allclose(42.0, m.interp_by_coord(coords_dict, method="nearest"))


def test_interp_methods():
    m = Map.create(npix=(3, 3))
    m.data += np.arange(9).reshape((3, 3))

    actual = m.interp_by_coord({"lon": 0.07, "lat": 0.03}, method="linear")
    assert_allclose(actual, 4.2)

    actual = m.interp_by_coord({"lon": 0.07, "lat": 0.03}, method="nearest")
    assert_allclose(actual, 3.0)


def test_wcsndmap_interp_by_coord_fill_value():
    # Introduced in https://github.com/gammapy/gammapy/pull/1559/files
    m = Map.create(npix=(20, 10))
    m.data += 42
    # With `fill_value` one should be able to control what gets filled
    assert_allclose(m.interp_by_coord((99, 0), fill_value=99), 99)
    # Default is to extrapolate
    assert_allclose(m.interp_by_coord((99, 0)), 42)


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), wcs_test_geoms
)
@pytest.mark.parametrize("keepdims", [True, False])
def test_wcsndmap_sum_over_axes(npix, binsz, frame, proj, skydir, axes, keepdims):
    geom = WcsGeom.create(npix=npix, binsz=binsz, proj=proj, frame=frame, axes=axes)
    m = WcsNDMap(geom)
    coords = m.geom.get_coord()
    m.fill_by_coord(coords, coords[0].value)
    msum = m.sum_over_axes(keepdims=keepdims)

    if m.geom.is_regular:
        assert_allclose(np.nansum(m.data), np.nansum(msum.data))


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_pad(npix, binsz, frame, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz, proj=proj, frame=frame, axes=axes)
    m = WcsNDMap(geom)
    m2 = m.pad(1, mode="constant", cval=2.2)
    if not geom.is_allsky:
        coords = m2.geom.get_coord()
        msk = m2.geom.contains(coords)
        coords = tuple([c[~msk] for c in coords])
        assert_allclose(m2.get_by_coord(coords), 2.2)
    m.pad(1, mode="interp", method="nearest")
    m.pad(1, mode="interp")


def test_wcsndmap_pad_cval():
    geom = WcsGeom.create(npix=(5, 5))
    m = WcsNDMap.from_geom(geom)

    cval = 1.1
    m_padded = m.pad(1, mode="constant", cval=cval)
    assert_allclose(m_padded.data[0, 0], cval)


def test_wcs_nd_map_pad_axis():
    axis = MapAxis.from_nodes([0, 1], unit="deg", name="axis")

    m = WcsNDMap.create(npix=3, axes=[axis])
    m.data += np.array([1, 2]).reshape((-1, 1, 1))

    m_pad = m.pad(axis_name="axis", pad_width=1, mode="edge")
    m_pad.data

    assert_allclose(m_pad.data[:, 1, 1], [1, 1, 2, 2])


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_crop(npix, binsz, frame, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz, proj=proj, frame=frame, axes=axes)
    m = WcsNDMap(geom)
    m.crop(1)


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_downsample(npix, binsz, frame, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz, proj=proj, frame=frame, axes=axes)
    m = WcsNDMap(geom, unit="m2")
    # Check whether we can downsample
    if np.all(np.mod(geom.npix[0], 2) == 0) and np.all(np.mod(geom.npix[1], 2) == 0):
        m2 = m.downsample(2, preserve_counts=True)
        assert_allclose(np.nansum(m.data), np.nansum(m2.data))
        assert m.unit == m2.unit


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), wcs_test_geoms
)
def test_wcsndmap_upsample(npix, binsz, frame, proj, skydir, axes):
    geom = WcsGeom.create(npix=npix, binsz=binsz, proj=proj, frame=frame, axes=axes)
    m = WcsNDMap(geom, unit="m2")
    m2 = m.upsample(2, preserve_counts=True)
    assert_allclose(np.nansum(m.data), np.nansum(m2.data))
    assert m.unit == m2.unit


def test_wcsndmap_upsample_axis():
    axis = MapAxis.from_edges([1, 2, 3, 4], name="test")
    geom = WcsGeom.create(npix=(2, 2), axes=[axis])
    test_nodes = np.arange(3)
    test_data = test_nodes.reshape(3, 1, 1)
    spatial_data = np.zeros((2, 2))
    data = spatial_data + 0.5 * test_data
    m = WcsNDMap(geom, unit="m2", data=data)

    m2 = m.upsample(2, preserve_counts=True, axis_name="test")
    assert m2.data.shape == (6, 2, 2)
    assert_allclose(m.data.sum(), m2.data.sum())

    assert_allclose(m2.data[:, 0, 0], [0, 0, 0.25, 0.25, 0.5, 0.5])


def test_wcsndmap_downsample_axis():
    axis = MapAxis.from_edges([1, 2, 3, 4, 5], name="test")
    geom = WcsGeom.create(npix=(4, 4), axes=[axis])
    m = WcsNDMap(geom, unit="m2")
    m.data += 1

    m2 = m.downsample(2, preserve_counts=True, axis_name="test")
    assert m2.data.shape == (2, 4, 4)


def test_wcsndmap_resample_axis():
    axis_1 = MapAxis.from_edges([1, 2, 3, 4, 5], name="test-1")
    axis_2 = MapAxis.from_edges([1, 2, 3, 4], name="test-2")

    geom = WcsGeom.create(npix=(7, 6), axes=[axis_1, axis_2])
    m = WcsNDMap(geom, unit="m2")
    m.data += 1

    new_axis = MapAxis.from_edges([1, 3, 5], name="test-1")
    m2 = m.resample_axis(axis=new_axis)
    assert m2.data.shape == (3, 2, 6, 7)
    assert_allclose(m2.data, 2)

    # Test without all interval covered
    new_axis = MapAxis.from_edges([2, 3], name="test-1")
    m3 = m.resample_axis(axis=new_axis)
    assert m3.data.shape == (3, 1, 6, 7)
    assert_allclose(m3.data, 1)


def test_wcsndmap_resample_axis_logical_and():
    axis_1 = MapAxis.from_edges([1, 2, 3, 4, 5], name="test-1")

    geom = WcsGeom.create(npix=(2, 2), axes=[axis_1])
    m = WcsNDMap(geom, dtype=bool)
    m.data[:, :, :] = True
    m.data[0, 0, 0] = False

    new_axis = MapAxis.from_edges([1, 3, 5], name="test-1")
    m2 = m.resample_axis(axis=new_axis, ufunc=np.logical_and)
    assert_allclose(m2.data[0, 0, 0], False)
    assert_allclose(m2.data[1, 0, 0], True)


def test_coadd_unit():
    geom = WcsGeom.create(npix=(10, 10), binsz=1, proj="CAR", frame="galactic")
    m1 = WcsNDMap(geom, data=np.ones((10, 10)), unit="m2")
    m2 = WcsNDMap(geom, data=np.ones((10, 10)), unit="cm2")

    m1.coadd(m2)

    assert_allclose(m1.data, 1.0001)


@pytest.mark.parametrize("kernel", ["gauss", "box", "disk"])
def test_smooth(kernel):
    axes = [
        MapAxis(np.logspace(0.0, 3.0, 3), interp="log"),
        MapAxis(np.logspace(1.0, 3.0, 4), interp="lin"),
    ]
    geom = WcsGeom.create(
        npix=(10, 10), binsz=1, proj="CAR", frame="galactic", axes=axes
    )
    m = WcsNDMap(geom, data=np.ones(geom.data_shape), unit="m2")

    desired = m.data.sum()
    smoothed = m.smooth(0.2 * u.deg, kernel)
    actual = smoothed.data.sum()
    assert_allclose(actual, desired)
    assert smoothed.data.dtype == float


@pytest.mark.parametrize("mode", ["partial", "strict", "trim"])
def test_make_cutout(mode):
    pos = SkyCoord(0, 0, unit="deg", frame="galactic")
    geom = WcsGeom.create(
        npix=(10, 10), binsz=1, skydir=pos, proj="CAR", frame="galactic", axes=axes2
    )
    m = WcsNDMap(geom, data=np.ones((3, 2, 10, 10)), unit="m2")
    cutout = m.cutout(position=pos, width=(2.0, 3.0) * u.deg, mode=mode)
    actual = cutout.data.sum()
    assert_allclose(actual, 36.0)
    assert_allclose(cutout.geom.shape_axes, m.geom.shape_axes)
    assert_allclose(cutout.geom.width.to_value("deg"), [[2.0], [3.0]])


def test_convolve_vs_smooth():
    axes = [
        MapAxis(np.logspace(0.0, 3.0, 3), interp="log"),
        MapAxis(np.logspace(1.0, 3.0, 4), interp="lin"),
    ]

    binsz = 0.05 * u.deg
    m = WcsNDMap.create(binsz=binsz, width=1.05 * u.deg, axes=axes)
    m.data[:, :, 10, 10] = 1.0

    desired = m.smooth(kernel="gauss", width=0.5 * u.deg, mode="constant")
    gauss = Gaussian2DKernel(10).array
    actual = m.convolve(kernel=gauss)
    assert_allclose(actual.data, desired.data, rtol=1e-3)


@requires_data()
def test_convolve_nd():
    energy_axis = MapAxis.from_edges(
        np.logspace(-1.0, 1.0, 4), unit="TeV", name="energy_true"
    )
    geom = WcsGeom.create(binsz=0.02 * u.deg, width=4.0 * u.deg, axes=[energy_axis])
    m = Map.from_geom(geom)
    m.fill_by_coord([[0.2, 0.4], [-0.1, 0.6], [0.5, 3.6]])

    psf = PSFMap.from_gauss(energy_axis, sigma=[0.1, 0.2, 0.3] * u.deg)
    psf_kernel = psf.get_psf_kernel(geom=geom, max_radius=1 * u.deg)

    assert psf_kernel.psf_kernel_map.data.shape == (3, 101, 101)

    mc = m.convolve(psf_kernel)
    assert_allclose(mc.data.sum(axis=(1, 2)), [0, 1, 1], atol=1e-5)

    kernel_2d = Box2DKernel(3, mode="center")
    kernel_2d.normalize("peak")
    mc = m.convolve(kernel_2d.array)
    assert_allclose(mc.data[0, :, :].sum(), 0, atol=1e-5)
    assert_allclose(mc.data[1, :, :].sum(), 9, atol=1e-5)

    kernel_2d = Gaussian2DKernel(15, mode="center")
    kernel_2d.normalize("peak")
    mc_full = m.convolve(kernel_2d.array, mode="full")
    mc_same = m.convolve(kernel_2d.array, mode="same")
    coords = [
        [0.2, 0.1, 0.4, 0.44, -1.3],
        [-0.1, -0.13, 0.6, 0.57, 0.91],
        [0.5, 0.5, 3.6, 3.6, 0.5],
    ]
    values_full = mc_full.get_by_coord(coords)
    values_same = mc_same.get_by_coord(coords)

    assert mc_same.data.shape == (3, 200, 200)
    assert mc_full.data.shape == (3, 320, 320)
    assert_allclose(values_full, values_same, rtol=1e-5)


def test_convolve_pixel_scale_error():
    m = WcsNDMap.create(binsz=0.05 * u.deg, width=5 * u.deg)
    kgeom = WcsGeom.create(binsz=0.04 * u.deg, width=0.5 * u.deg)

    kernel = PSFKernel.from_gauss(kgeom, sigma=0.1 * u.deg, max_radius=1.5 * u.deg)

    with pytest.raises(ValueError):
        m.convolve(kernel)


def test_convolve_kernel_size_error():
    axis_1 = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=2)
    axis_2 = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=3)

    m = WcsNDMap.create(binsz=0.05 * u.deg, width=5 * u.deg, axes=[axis_1])

    kgeom = WcsGeom.create(binsz=0.05 * u.deg, width=0.5 * u.deg, axes=[axis_2])
    kernel = PSFKernel.from_gauss(kgeom, sigma=0.1 * u.deg, max_radius=1.5 * u.deg)

    with pytest.raises(ValueError):
        m.convolve(kernel)


def test_plot():
    axis = MapAxis([0, 1], node_type="edges")
    m = WcsNDMap.create(binsz=0.1 * u.deg, width=1 * u.deg, axes=[axis])
    with mpl_plot_check():
        m.plot(add_cbar=True)


def test_plot_grid():
    axis = MapAxis([0, 1, 2], node_type="edges")
    m = WcsNDMap.create(binsz=0.1 * u.deg, width=1 * u.deg, axes=[axis])
    with mpl_plot_check():
        m.plot_grid()


def test_plot_allsky():
    axis = MapAxis([0, 1], node_type="edges")
    m = WcsNDMap.create(binsz=10 * u.deg, axes=[axis])
    with mpl_plot_check():
        m.plot()


def test_plot_nan():
    m = Map.create(width=10, binsz=1)
    m.data += np.nan
    with mpl_plot_check():
        m.plot(add_cbar=False)


def test_get_spectrum():
    axis = MapAxis.from_bounds(1, 10, nbin=3, unit="TeV", name="energy")

    geom = WcsGeom.create(
        skydir=(0, 0), width=(2.5, 2.5), binsz=0.5, axes=[axis], frame="galactic"
    )

    m = Map.from_geom(geom)
    m.data += 1

    center = SkyCoord(0, 0, frame="galactic", unit="deg")
    region = CircleSkyRegion(center=center, radius=1 * u.deg)

    spec = m.get_spectrum(region=region)
    assert_allclose(spec.data.squeeze(), [13.0, 13.0, 13.0])

    spec = m.get_spectrum(region=region, func=np.mean)
    assert_allclose(spec.data.squeeze(), [1.0, 1.0, 1.0])

    spec = m.get_spectrum()
    assert isinstance(spec.geom.region, RectangleSkyRegion)

    region = PointSkyRegion(center)
    spec = m.get_spectrum(region=region)
    assert_allclose(spec.data.squeeze(), [1.0, 1.0, 1.0])


def test_get_spectrum_type():
    axis = MapAxis.from_bounds(1, 10, nbin=3, unit="TeV", name="energy")

    geom = WcsGeom.create(
        skydir=(0, 0), width=(2.5, 2.5), binsz=0.5, axes=[axis], frame="galactic"
    )

    m_int = Map.from_geom(geom, dtype="int")
    m_int.data += 1

    m_bool = Map.from_geom(geom, dtype="bool")
    m_bool.data += True

    center = SkyCoord(0, 0, frame="galactic", unit="deg")
    region = CircleSkyRegion(center=center, radius=1 * u.deg)

    spec_int = m_int.get_spectrum(region=region)
    assert spec_int.data.dtype == np.dtype("int")
    assert_allclose(spec_int.data.squeeze(), [13, 13, 13])

    spec_bool = m_bool.get_spectrum(region=region, func=np.any)
    assert spec_bool.data.dtype == np.dtype("bool")
    assert_allclose(spec_bool.data.squeeze(), [1, 1, 1])


def test_get_spectrum_weights():
    axis = MapAxis.from_bounds(1, 10, nbin=3, unit="TeV", name="energy")

    geom = WcsGeom.create(
        skydir=(0, 0), width=(2.5, 2.5), binsz=0.5, axes=[axis], frame="galactic"
    )

    m_int = Map.from_geom(geom, dtype="int")
    m_int.data += 1

    weights = Map.from_geom(geom, dtype="bool")
    weights.data[:, 2, 2] = True

    bad_weights = Map.from_geom(geom.to_image(), dtype="bool")

    center = SkyCoord(0, 0, frame="galactic", unit="deg")
    region = CircleSkyRegion(center=center, radius=1 * u.deg)

    spec_int = m_int.get_spectrum(region=region, weights=weights)
    assert spec_int.data.dtype == np.dtype("int")
    assert_allclose(spec_int.data.squeeze(), [1, 1, 1])

    with pytest.raises(ValueError):
        m_int.get_spectrum(region=region, weights=bad_weights)


def get_npred_map():
    position = SkyCoord(0.0, 0.0, frame="galactic", unit="deg")
    energy_axis = MapAxis.from_bounds(
        1, 100, nbin=30, unit="TeV", name="energy_true", interp="log"
    )

    exposure = Map.create(
        binsz=0.02,
        map_type="wcs",
        skydir=position,
        width="2 deg",
        axes=[energy_axis],
        frame="galactic",
        unit="cm2 s",
    )

    spatial_model = GaussianSpatialModel(
        lon_0="0.015 deg", lat_0="-0.037 deg", sigma="0.2 deg", frame="galactic"
    )
    spectral_model = PowerLawSpectralModel(amplitude="1e-11 cm-2 s-1 TeV-1")
    skymodel = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

    exposure.data = 1e14 * np.ones(exposure.data.shape)
    evaluator = MapEvaluator(model=skymodel, exposure=exposure)

    npred = evaluator.compute_npred()
    return evaluator, npred


def test_map_sampling():
    eval, npred = get_npred_map()

    nmap = WcsNDMap(geom=eval.geom, data=npred.data)
    coords = nmap.sample_coord(n_events=2, random_state=0)

    assert len(coords["lon"]) == 2
    assert_allclose(coords.skycoord.icrs.ra.deg, [266.204197, 266.451241], rtol=1e-5)
    assert_allclose(coords.skycoord.icrs.dec.deg, [-28.862369, -29.075469], rtol=1e-5)
    assert_allclose(coords["energy_true"].data, [2.363293, 2.342388], rtol=1e-5)

    assert coords["lon"].unit == "deg"
    assert coords["lat"].unit == "deg"
    assert coords["energy_true"].unit == "TeV"


def test_map_interp_one_bin():
    m = WcsNDMap.create(npix=(2, 1))
    m.data = np.array([[1, 2]])

    coords = {"lon": 0, "lat": [0, 0]}
    data = m.interp_by_coord(coords)

    assert data.shape == (2,)
    assert_allclose(data, 1.5)


def test_sum_over_axes():
    # Check summing over a specific axis
    ax1 = MapAxis.from_nodes([1, 2, 3, 4], name="ax1")
    ax2 = MapAxis.from_nodes([5, 6, 7], name="ax2")
    ax3 = MapAxis.from_nodes([8, 9], name="ax3")
    geom = WcsGeom.create(npix=(5, 5), axes=[ax1, ax2, ax3])
    m1 = Map.from_geom(geom=geom)
    m1.data = np.ones(m1.data.shape)
    m2 = m1.sum_over_axes(axes_names=["ax1", "ax3"], keepdims=True)

    assert_allclose(m2.geom.data_shape, (1, 3, 1, 5, 5))
    assert_allclose(m2.data[0][0][0][0][0], 8.0)

    m3 = m1.sum_over_axes(axes_names=["ax3", "ax2"], keepdims=False)
    assert_allclose(m3.geom.data_shape, (4, 5, 5))
    assert_allclose(m3.data[0][0][0], 6.0)


def test_reduce():
    # Check summing over a specific axis
    ax1 = MapAxis.from_nodes([1, 2, 3, 4], name="ax1")
    ax2 = MapAxis.from_nodes([5, 6, 7], name="ax2")
    ax3 = MapAxis.from_nodes([8, 9], name="ax3")
    geom = WcsGeom.create(npix=(5, 5), axes=[ax1, ax2, ax3])
    m1 = Map.from_geom(geom=geom)
    m1.data = np.ones(m1.data.shape)
    m2 = m1.reduce(axis_name="ax1", keepdims=True)

    assert_allclose(m2.geom.data_shape, (2, 3, 1, 5, 5))
    assert_allclose(m2.data[0][0][0][0][0], 4.0)

    m3 = m1.reduce(axis_name="ax1", keepdims=False)
    assert_allclose(m3.geom.data_shape, (2, 3, 5, 5))
    assert_allclose(m3.data[0][0][0][0], 4.0)


def test_to_cube():
    ax1 = MapAxis.from_nodes([1, 2, 3, 4], name="ax1")
    ax2 = MapAxis.from_edges([5, 6], name="ax2")
    ax3 = MapAxis.from_edges([8, 9], name="ax3")
    geom = WcsGeom.create(npix=(5, 5), axes=[ax1])
    m1 = Map.from_geom(geom=geom, data=np.ones(geom.data_shape))
    m2 = m1.to_cube([ax2, ax3])
    assert_allclose(m2.geom.data_shape, (1, 1, 4, 5, 5))

    # test that more than one bin fails
    ax4 = MapAxis.from_edges([8, 9, 10], name="ax4")
    with pytest.raises(ValueError):
        m1.to_cube([ax4])


def test_stack_unit_handling():
    m = WcsNDMap.create(npix=(3, 3), unit="m2 s")
    m.data += 1

    m_other = WcsNDMap.create(npix=(3, 3), unit="cm2 s")
    m_other.data += 1

    m.stack(m_other)

    assert_allclose(m.data, 1.0001)


def test_binary_erode():
    geom = WcsGeom.create(binsz=0.02, width=2 * u.deg)
    mask = geom.region_mask("icrs;circle(0, 0, 1)")

    mask = mask.binary_erode(width=0.2 * u.deg, kernel="disk", use_fft=False)
    assert_allclose(mask.data.sum(), 4832)

    mask = mask.binary_erode(width=0.2 * u.deg, kernel="box", use_fft=True)
    # Due to fft noise the result is not exact here.
    # See https://github.com/gammapy/gammapy/issues/3662
    assert_allclose(mask.data.sum(), 3372, atol=20)


def test_binary_dilate():
    geom = WcsGeom.create(binsz=0.02, width=2 * u.deg)
    mask = geom.region_mask("icrs;circle(0, 0, 0.8)")

    mask = mask.binary_dilate(width=0.2 * u.deg, kernel="disk", use_fft=False)
    assert_allclose(mask.data.sum(), 8048)

    mask = mask.binary_dilate(width=(10, 10), kernel="box")
    # Due to fft noise the result is not exact here.
    # See https://github.com/gammapy/gammapy/issues/3662
    assert_allclose(mask.data.sum(), 9203, atol=20)


def test_binary_dilate_erode_3d():
    axis = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=2)
    geom = WcsGeom.create(
        binsz=0.02,
        width=(2, 2),
        frame="icrs",
        axes=[axis],
    )

    mask = Map.from_geom(geom=geom, dtype=bool)
    mask.data |= True

    mask_fit = mask.binary_erode(width=(0.3 * u.deg, 0.1 * u.deg))
    assert np.sum(mask_fit.data) == 9800

    mask = geom.boundary_mask(width=(0.3 * u.deg, 0.1 * u.deg))
    mask = mask.binary_dilate(width=(0.6 * u.deg, 0.2 * u.deg))
    assert np.sum(mask.data) == np.prod(mask.data.shape)


def test_memory_usage():
    geom = WcsGeom.create()
    assert geom.data_nbytes().unit == u.MB
    assert_allclose(geom.data_nbytes(dtype="float32").value, 1.0368)
    assert_allclose(geom.data_nbytes(dtype="b").value, 0.2592)


def test_double_cutout():
    # regression test for https://github.com/gammapy/gammapy/issues/3368
    m = Map.create(width="10 deg")
    m.data = np.arange(10_000, dtype="float")

    position = SkyCoord("1d", "1d")
    m_c = m.cutout(position=position, width="3 deg")
    m_cc = m_c.cutout(position=position, width="2 deg")

    m_new = Map.create(width="10 deg")
    m_new.stack(m_cc)
    m_c_new = m_new.cutout(position=position, width="2 deg")
    np.testing.assert_allclose(m_c_new.data, m_cc.data)

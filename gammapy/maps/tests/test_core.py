# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.units import Quantity, Unit
from gammapy.maps import HpxGeom, HpxNDMap, Map, MapAxis, TimeMapAxis, WcsGeom, WcsNDMap
from gammapy.utils.testing import mpl_plot_check

pytest.importorskip("healpy")

map_axes = [
    MapAxis.from_bounds(1.0, 10.0, 3, interp="log", name="energy"),
    MapAxis.from_bounds(0.1, 1.0, 4, interp="log", name="time"),
]

mapbase_args = [
    (0.1, 10.0, "wcs", SkyCoord(0.0, 30.0, unit="deg"), None, ""),
    (0.1, 10.0, "wcs", SkyCoord(0.0, 30.0, unit="deg"), map_axes[:1], ""),
    (0.1, 10.0, "wcs", SkyCoord(0.0, 30.0, unit="deg"), map_axes, "m^2"),
    (0.1, 10.0, "hpx", SkyCoord(0.0, 30.0, unit="deg"), None, ""),
    (0.1, 10.0, "hpx", SkyCoord(0.0, 30.0, unit="deg"), map_axes[:1], ""),
    (0.1, 10.0, "hpx", SkyCoord(0.0, 30.0, unit="deg"), map_axes, "s^2"),
]

mapbase_args_with_axes = [_ for _ in mapbase_args if _[4] is not None]


@pytest.mark.parametrize(
    ("binsz", "width", "map_type", "skydir", "axes", "unit"), mapbase_args
)
def test_map_create(binsz, width, map_type, skydir, axes, unit):
    m = Map.create(
        binsz=binsz, width=width, map_type=map_type, skydir=skydir, axes=axes, unit=unit
    )
    assert m.unit == unit


@pytest.mark.parametrize(
    ("binsz", "width", "map_type", "skydir", "axes", "unit"), mapbase_args_with_axes
)
def test_map_copy(binsz, width, map_type, skydir, axes, unit):
    m = Map.create(
        binsz=binsz, width=width, map_type=map_type, skydir=skydir, axes=axes, unit=unit
    )

    m_copy = m.copy()
    assert repr(m) == repr(m_copy)

    m_copy = m.copy(unit="cm-2 s-1")
    assert m_copy.unit == "cm-2 s-1"
    assert m_copy.unit is not m.unit

    m_copy = m.copy(meta={"is_copy": True})
    assert m_copy.meta["is_copy"]
    assert m_copy.meta is not m.meta

    m_copy = m.copy(data=42 * np.ones(m.data.shape))
    assert m_copy.data[(0,) * m_copy.data.ndim] == 42
    assert m_copy.data is not m.data


def test_map_from_geom():
    geom = WcsGeom.create(binsz=1.0, width=10.0)
    m = Map.from_geom(geom)
    assert isinstance(m, WcsNDMap)
    assert m.geom.is_image

    geom = HpxGeom.create(binsz=1.0, width=10.0)
    m = Map.from_geom(geom)
    assert isinstance(m, HpxNDMap)
    assert m.geom.is_image


@pytest.mark.parametrize(
    ("binsz", "width", "map_type", "skydir", "axes", "unit"), mapbase_args_with_axes
)
def test_map_get_image_by_coord(binsz, width, map_type, skydir, axes, unit):
    m = Map.create(
        binsz=binsz, width=width, map_type=map_type, skydir=skydir, axes=axes, unit=unit
    )
    m.data = np.arange(m.data.size, dtype=float).reshape(m.data.shape)

    coords = (3.456, 0.1234)[: len(m.geom.axes)]

    m_image = m.get_image_by_coord(coords)

    im_geom = m.geom.to_image()
    skycoord = im_geom.get_coord().skycoord

    m_vals = m.get_by_coord((skycoord,) + coords)
    assert_equal(m_image.data, m_vals)


@pytest.mark.parametrize(
    ("binsz", "width", "map_type", "skydir", "axes", "unit"), mapbase_args_with_axes
)
def test_map_get_image_by_pix(binsz, width, map_type, skydir, axes, unit):
    m = Map.create(
        binsz=binsz, width=width, map_type=map_type, skydir=skydir, axes=axes, unit=unit
    )
    pix = (1.2345, 0.1234)[: len(m.geom.axes)]
    m_image = m.get_image_by_pix(pix)

    im_geom = m.geom.to_image()
    idx = im_geom.get_idx()
    m_vals = m.get_by_pix(idx + pix)
    assert_equal(m_image.data, m_vals)


@pytest.mark.parametrize(
    ("binsz", "width", "map_type", "skydir", "axes", "unit"), mapbase_args_with_axes
)
def test_map_slice_by_idx(binsz, width, map_type, skydir, axes, unit):
    m = Map.create(
        binsz=binsz, width=width, map_type=map_type, skydir=skydir, axes=axes, unit=unit
    )
    data = np.arange(m.data.size, dtype=float)
    m.data = data.reshape(m.data.shape)

    # Test none slicing
    sliced = m.slice_by_idx({})
    assert_equal(m.geom.shape_axes, sliced.geom.shape_axes)

    slices = {"energy": slice(0, 1), "time": slice(0, 2)}
    sliced = m.slice_by_idx(slices)
    assert not sliced.geom.is_image
    slices = tuple([slices[ax.name] for ax in m.geom.axes])
    assert_equal(m.data[slices[::-1]], sliced.data)
    assert sliced.data.base is data

    slices = {"energy": 0, "time": 1}
    sliced = m.slice_by_idx(slices)
    assert sliced.geom.is_image
    slices = tuple([slices[ax.name] for ax in m.geom.axes])
    assert_equal(m.data[slices[::-1]], sliced.data)
    assert sliced.data.base is data


@pytest.mark.parametrize("map_type", ["wcs", "hpx"])
def test_map_meta_read_write(map_type):
    meta = {"user": "test"}

    m = Map.create(
        binsz=0.1,
        width=10.0,
        map_type=map_type,
        skydir=SkyCoord(0.0, 30.0, unit="deg"),
        meta=meta,
    )

    hdulist = m.to_hdulist(hdu="COUNTS")
    header = hdulist["COUNTS"].header

    assert header["META"] == '{"user": "test"}'

    m2 = Map.from_hdulist(hdulist)
    assert m2.meta == meta


@pytest.mark.parametrize("map_type", ["wcs", "hpx"])
def test_map_time_axis_read_write(map_type):
    time_axis = TimeMapAxis(
        edges_min=[0, 2, 4] * u.d,
        edges_max=[1, 3, 5] * u.d,
        reference_time="2000-01-01",
    )

    energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=5)

    m = Map.create(
        binsz=0.1,
        width=10.0,
        map_type=map_type,
        skydir=SkyCoord(0.0, 30.0, unit="deg"),
        axes=[energy_axis, time_axis],
    )

    hdulist = m.to_hdulist(hdu="COUNTS")

    m2 = Map.from_hdulist(hdulist)

    time_axis_new = m2.geom.axes["time"]
    assert time_axis_new == time_axis
    assert time_axis.reference_time.scale == "utc"
    assert time_axis_new.reference_time.scale == "tt"


unit_args = [("wcs", "s"), ("wcs", ""), ("wcs", Unit("sr")), ("hpx", "m^2")]


@pytest.mark.parametrize(("map_type", "unit"), unit_args)
def test_map_quantity(map_type, unit):
    m = Map.create(binsz=0.1, width=10.0, map_type=map_type, unit=unit)

    # This is to test if default constructor with no unit performs as expected
    if unit is None:
        unit = ""
    assert m.quantity.unit == Unit(unit)

    m.quantity = Quantity(np.ones_like(m.data), "m2")
    assert m.unit == "m2"

    m1 = m.__class__(geom=m.geom, data=m.quantity)
    assert m1.unit == "m2"
    assert_allclose(m1.data, m.data)


@pytest.mark.parametrize(("map_type", "unit"), unit_args)
def test_map_unit_read_write(map_type, unit):
    m = Map.create(binsz=0.1, width=10.0, map_type=map_type, unit=unit)

    hdu_list = m.to_hdulist(hdu="COUNTS")
    header = hdu_list["COUNTS"].header

    assert Unit(header["BUNIT"]) == Unit(unit)

    m2 = Map.from_hdulist(hdu_list)
    assert m2.unit == unit


@pytest.mark.parametrize(("map_type", "unit"), unit_args)
def test_map_repr(map_type, unit):
    m = Map.create(binsz=0.1, width=10.0, map_type=map_type, unit=unit)
    assert m.__class__.__name__ in repr(m)


def test_map_properties():
    # Test default values and types of all map properties,
    # as well as the behaviour for the property get and set.

    m = Map.create(npix=(2, 1))

    assert isinstance(m.unit, u.CompositeUnit)
    assert m._unit == u.one
    m._unit = u.Unit("cm-2 s-1")
    assert m.unit.to_string() == "1 / (cm2 s)"

    assert isinstance(m.meta, dict)
    m.meta = {"spam": 42}
    assert isinstance(m.meta, dict)

    # The rest of the tests are for the `data` property

    assert isinstance(m.data, np.ndarray)
    assert m.data.dtype == np.float32
    assert m.data.shape == (1, 2)
    assert_equal(m.data, 0)

    # Assigning an array of matching shape stores it away
    data = np.ones((1, 2))
    m.data = data
    assert m.data is data

    # In-place modification += should work as expected
    m.data = np.array([[42, 43]])
    data = m.data
    m.data += 1
    assert m.data is data
    assert_equal(m.data, [[43, 44]])

    # Assigning to a slice of the map data should work as expected
    data = m.data
    m.data[:, :1] = 99
    assert m.data is data
    assert_equal(m.data, [[99, 44]])

    # Assigning something that doesn't match raises an appropriate error
    with pytest.raises(ValueError):
        m.data = np.ones((1, 3))


map_arithmetics_args = [("wcs"), ("hpx")]


@pytest.mark.parametrize(("map_type"), map_arithmetics_args)
def test_map_arithmetics(map_type):

    m1 = Map.create(binsz=0.1, width=1.0, map_type=map_type, skydir=(0, 0), unit="m2")

    m2 = Map.create(binsz=0.1, width=1.0, map_type=map_type, skydir=(0, 0), unit="m2")
    m2.data += 1.0

    # addition
    m1 += 1 * u.cm**2
    assert m1.unit == u.Unit("m2")
    assert_allclose(m1.data, 1e-4)

    m3 = m1 + m2
    assert m3.unit == u.Unit("m2")
    assert_allclose(m3.data, 1.0001)

    # subtraction
    m3 -= 1 * u.cm**2
    assert m3.unit == u.Unit("m2")
    assert_allclose(m3.data, 1.0)

    m3 = m2 - m1
    assert m3.unit == u.Unit("m2")
    assert_allclose(m3.data, 0.9999)

    m4 = Map.create(binsz=0.1, width=1.0, map_type=map_type, skydir=(0, 0), unit="s")
    m4.data += 1.0

    # multiplication
    m1 *= 1e4
    assert m1.unit == u.Unit("m2")
    assert_allclose(m1.data, 1)

    m5 = m2 * m4
    assert m5.unit == u.Unit("m2s")
    assert_allclose(m5.data, 1)

    # division
    m5 /= 10 * u.s
    assert m5.unit == u.Unit("m2")
    assert_allclose(m5.data, 0.1)

    # check unit consistency
    with pytest.raises(u.UnitConversionError):
        m5 += 1 * u.W

    m1.data *= 0.0
    m1._unit = u.one
    m1 += 4
    assert m1.unit == u.Unit("")
    assert_allclose(m1.data, 4)

    lt_m2 = m2 < 1.5 * u.m**2
    assert lt_m2.data.dtype == bool
    assert_allclose(lt_m2, True)

    le_m2 = m2 <= 10000 * u.cm**2
    assert_allclose(le_m2, True)

    gt_m2 = m2 > 15000 * u.cm**2
    assert_allclose(gt_m2, False)

    ge_m2 = m2 >= m2
    assert_allclose(ge_m2, True)

    eq_m2 = m2 == 500 * u.cm**2
    assert_allclose(eq_m2, False)

    ne_m2 = m2 != 500 * u.cm**2
    assert_allclose(ne_m2, True)


def test_boolean_arithmetics():
    m_1 = Map.create(binsz=1, width=2)
    m_1.data = True

    m_2 = Map.create(binsz=1, width=2)
    m_2.data = False

    m_and = m_1 & m_2
    assert not np.any(m_and.data)

    m_or = m_1 | m_2
    assert np.all(m_or.data)

    m_not = ~m_2
    assert np.all(m_not.data)

    m_xor = m_1 ^ m_1
    assert not np.any(m_xor.data)


def test_arithmetics_inconsistent_geom():
    m_wcs = Map.create(binsz=0.1, width=1.0)
    m_wcs_incorrect = Map.create(binsz=0.1, width=2.0)

    with pytest.raises(ValueError):
        m_wcs += m_wcs_incorrect

    m_hpx = Map.create(binsz=0.1, width=1.0, map_type="hpx")
    with pytest.raises(ValueError):
        m_wcs += m_hpx


# TODO: correct serialization for lin axis for energy
# map_serialization_args = [("log"), ("lin")]

map_serialization_args = [("log")]


@pytest.mark.parametrize(("interp"), map_serialization_args)
def test_arithmetics_after_serialization(tmp_path, interp):
    axis = MapAxis.from_bounds(
        1.0, 10.0, 3, interp=interp, name="energy", node_type="center", unit="TeV"
    )
    m_wcs = Map.create(binsz=0.1, width=1.0, map_type="wcs", skydir=(0, 0), axes=[axis])
    m_wcs += 1

    m_wcs.write(tmp_path / "tmp.fits")
    m_wcs_serialized = Map.read(tmp_path / "tmp.fits")

    m_wcs += m_wcs_serialized

    assert_allclose(m_wcs.data, 2.0)


def test_set_scalar():
    m = Map.create(width=1)
    m.data = 1
    assert m.data.shape == (10, 10)
    assert_allclose(m.data, 1)


def test_interp_to_geom():
    energy = MapAxis.from_energy_bounds("1 TeV", "300 TeV", nbin=5, name="energy")
    energy_target = MapAxis.from_energy_bounds(
        "1 TeV", "300 TeV", nbin=7, name="energy"
    )
    value = 30
    coords = {"skycoord": SkyCoord("0 deg", "0 deg"), "energy": energy_target.center[3]}

    # WcsNDMap
    geom_wcs = WcsGeom.create(
        npix=(5, 3), proj="CAR", binsz=60, axes=[energy], skydir=(0, 0)
    )
    wcs_map = Map.from_geom(geom_wcs, unit="")
    wcs_map.data = value * np.ones(wcs_map.data.shape)

    wcs_geom_target = WcsGeom.create(
        skydir=(0, 0), width=(10, 10), binsz=0.1 * u.deg, axes=[energy_target]
    )
    interp_wcs_map = wcs_map.interp_to_geom(wcs_geom_target, method="linear")

    assert_allclose(interp_wcs_map.get_by_coord(coords)[0], value, atol=1e-7)
    assert isinstance(interp_wcs_map, WcsNDMap)
    assert interp_wcs_map.geom == wcs_geom_target

    # HpxNDMap
    geom_hpx = HpxGeom.create(binsz=60, axes=[energy], skydir=(0, 0))
    hpx_map = Map.from_geom(geom_hpx, unit="")
    hpx_map.data = value * np.ones(hpx_map.data.shape)

    hpx_geom_target = HpxGeom.create(
        skydir=(0, 0), width=10, binsz=0.1 * u.deg, axes=[energy_target]
    )
    interp_hpx_map = hpx_map.interp_to_geom(hpx_geom_target)

    assert_allclose(interp_hpx_map.get_by_coord(coords)[0], value, atol=1e-7)
    assert isinstance(interp_hpx_map, HpxNDMap)
    assert interp_hpx_map.geom == hpx_geom_target

    # Preserving the counts
    binsz = 0.2 * u.deg
    geom_initial = WcsGeom.create(
        skydir=(20, 20),
        width=(1, 1),
        binsz=0.2 * u.deg,
    )

    factor = 2
    test_map = Map.from_geom(geom_initial, unit="")
    test_map.data = value * np.eye(test_map.data.shape[0])
    geom_target = WcsGeom.create(
        skydir=(20, 20),
        width=(1.5, 1.5),
        binsz=binsz / factor,
    )
    new_map = test_map.interp_to_geom(
        geom_target, fill_value=0.0, method="nearest", preserve_counts=True
    )
    assert_allclose(new_map.data[8, 8], test_map.data[4, 4] / factor**2, rtol=1e-4)
    assert_allclose(new_map.data[0, 8], 0.0, rtol=1e-4)


def test_map_plot_mask():
    from regions import CircleSkyRegion

    skydir = SkyCoord(0, 0, frame="galactic", unit="deg")

    m_wcs = Map.create(
        map_type="wcs",
        binsz=0.02,
        skydir=skydir,
        width=2.0,
    )

    exclusion_region = CircleSkyRegion(
        center=SkyCoord(0.0, 0.0, unit="deg", frame="galactic"), radius=0.6 * u.deg
    )

    mask = ~m_wcs.geom.region_mask([exclusion_region])

    with mpl_plot_check():
        mask.plot_mask()


def test_reproject_2d():
    npix1 = 3
    geom1 = WcsGeom.create(npix=npix1, frame="icrs")
    geom1_large = WcsGeom.create(npix=npix1 + 5, frame="icrs")
    map1 = Map.from_geom(geom1, data=np.eye(npix1))

    factor = 10
    binsz = 0.5 / factor
    npix2 = 7 * factor
    geom2 = WcsGeom.create(
        skydir=SkyCoord(0.1, 0.1, unit=u.deg), binsz=binsz, npix=npix2, frame="galactic"
    )

    map1_repro = map1.reproject_to_geom(geom2, preserve_counts=True)
    assert_allclose(np.sum(map1_repro), np.sum(map1), rtol=1e-5)
    map1_new = map1_repro.reproject_to_geom(geom1_large, preserve_counts=True)
    assert_allclose(np.sum(map1_repro), np.sum(map1_new), rtol=1e-5)

    map1_repro = map1.reproject_to_geom(geom2, preserve_counts=False)
    map1_new = map1_repro.reproject_to_geom(geom1_large, preserve_counts=False)
    assert_allclose(
        np.sum(map1_repro * geom2.solid_angle()),
        np.sum(map1_new * geom1_large.solid_angle()),
        rtol=1e-3,
    )

    factor = 0.5
    binsz = 0.5 / factor
    npix = 7
    geom2 = WcsGeom.create(
        skydir=SkyCoord(0.1, 0.1, unit=u.deg), binsz=binsz, npix=npix, frame="galactic"
    )
    geom1_large = WcsGeom.create(npix=npix1 + 5, frame="icrs")

    map1_repro = map1.reproject_to_geom(geom2, preserve_counts=True)
    assert_allclose(np.sum(map1_repro), np.sum(map1), rtol=1e-5)
    map1_new = map1_repro.reproject_to_geom(geom1_large, preserve_counts=True)
    assert_allclose(np.sum(map1_repro), np.sum(map1_new), rtol=1e-3)

    map1_repro = map1.reproject_to_geom(geom2, preserve_counts=False)
    map1_new = map1_repro.reproject_to_geom(geom1_large, preserve_counts=False)
    assert_allclose(
        np.sum(map1_repro * geom2.solid_angle()),
        np.sum(map1_new * geom1_large.solid_angle()),
        rtol=1e-3,
    )


def test_resample_wcs_Wcs():
    npix1 = 3
    geom1 = WcsGeom.create(npix=npix1, frame="icrs")
    map1 = Map.from_geom(geom1, data=np.eye(npix1))

    geom2 = WcsGeom.create(
        skydir=SkyCoord(0.0, 0.0, unit=u.deg), binsz=0.5, npix=7, frame="galactic"
    )

    map2 = map1.resample(geom2, preserve_counts=False)
    assert_allclose(
        np.sum(map2 * geom2.solid_angle()),
        np.sum(map1 * geom1.solid_angle()),
        rtol=1e-3,
    )


def test_resample_weights():
    npix1 = 3
    geom1 = WcsGeom.create(npix=npix1, frame="icrs")
    map1 = Map.from_geom(geom1, data=np.eye(npix1))

    geom2 = WcsGeom.create(
        skydir=SkyCoord(0.0, 0.0, unit=u.deg), binsz=0.5, npix=7, frame="galactic"
    )

    map2 = map1.resample(geom2, weights=np.zeros(npix1), preserve_counts=False)
    assert np.sum(map2.data) == 0.0


def test_resample_downsample_wcs():
    geom_fine = WcsGeom.create(npix=(15, 15), frame="icrs")
    map_fine = Map.from_geom(geom_fine)
    map_fine.data = np.arange(225).reshape((15, 15))

    map_downsampled = map_fine.downsample(3)
    map_resampled = map_fine.resample(geom_fine.downsample(3), preserve_counts=True)

    assert_allclose(map_downsampled, map_resampled, rtol=1e-5)


def test_resample_downsample_axis():
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=12)
    geom_fine = WcsGeom.create(npix=(3, 3), frame="icrs", axes=[axis])
    map_fine = Map.from_geom(geom_fine)
    map_fine.data = np.arange(108).reshape((12, 3, 3))

    map_downsampled = map_fine.downsample(3, axis_name="energy", preserve_counts=False)
    map_resampled = map_fine.resample(
        geom_fine.downsample(3, axis_name="energy"), preserve_counts=False
    )

    assert_allclose(map_downsampled, map_resampled, rtol=1e-5)


def test_resample_wcs_hpx():
    geom1 = HpxGeom.create(nside=32, frame="icrs")
    map1 = Map.from_geom(geom1, data=1.0)
    geom2 = HpxGeom.create(
        skydir=SkyCoord(0.0, 0.0, unit=u.deg), nside=8, frame="galactic"
    )

    map2 = map1.resample(geom2, weights=map1.quantity, preserve_counts=False)
    assert_allclose(
        np.sum(map2 * geom2.solid_angle()),
        np.sum(map1 * geom1.solid_angle()),
        rtol=1e-3,
    )


def test_map_reproject_wcs_to_hpx():
    axis = MapAxis.from_bounds(
        1.0, 10.0, 3, interp="log", name="energy", node_type="center"
    )
    axis2 = MapAxis.from_bounds(
        1.0, 10.0, 8, interp="log", name="energy", node_type="center"
    )
    geom_wcs = WcsGeom.create(
        skydir=(0, 0), npix=(11, 11), binsz=10, axes=[axis], frame="galactic"
    )
    geom_hpx = HpxGeom.create(binsz=10, frame="galactic", axes=[axis2])

    data = np.arange(11 * 11 * 3).reshape(geom_wcs.data_shape)
    m = WcsNDMap(geom_wcs, data=data)

    m_r = m.reproject_to_geom(geom_hpx.to_image())
    actual = m_r.get_by_coord({"lon": 0, "lat": 0, "energy": [1.0, 3.16227766, 10.0]})
    assert_allclose(actual, [65.0, 186.0, 307.0], rtol=1e-3)

    with pytest.raises(TypeError):
        m.reproject_to_geom(geom_hpx)


def test_map_reproject_hpx_to_wcs():
    axis = MapAxis.from_bounds(
        1.0, 10.0, 3, interp="log", name="energy", node_type="center"
    )
    axis2 = MapAxis.from_bounds(
        1.0, 10.0, 8, interp="log", name="energy", node_type="center"
    )
    geom_wcs = WcsGeom.create(
        skydir=(0, 0), npix=(11, 11), binsz=10, axes=[axis], frame="galactic"
    )
    geom_wcs2 = WcsGeom.create(
        skydir=(0, 0), npix=(11, 11), binsz=10, axes=[axis2], frame="galactic"
    )
    geom_hpx = HpxGeom.create(binsz=10, frame="galactic", axes=[axis])

    data = np.arange(3 * 768).reshape(geom_hpx.data_shape)
    m = HpxNDMap(geom_hpx, data=data)

    m_r = m.reproject_to_geom(geom_wcs.to_image())
    actual = m_r.get_by_coord({"lon": 0, "lat": 0, "energy": [1.0, 3.16227766, 10.0]})
    assert_allclose(actual, [287.5, 1055.5, 1823.5], rtol=1e-3)

    m_r = m.reproject_to_geom(geom_wcs)
    actual = m_r.get_by_coord({"lon": 0, "lat": 0, "energy": [1.0, 3.16227766, 10.0]})
    assert_allclose(actual, [287.5, 1055.5, 1823.5], rtol=1e-3)

    with pytest.raises(TypeError):
        m.reproject_to_geom(geom_wcs2)


def test_map_reproject_wcs_to_wcs_with_axes():
    energy_nodes = np.arange(4) - 0.5
    time_nodes = np.arange(5) - 0.5

    axis1 = MapAxis(energy_nodes, interp="lin", name="energy", node_type="edges")
    axis2 = MapAxis(time_nodes, interp="lin", name="time", node_type="edges")
    geom_wcs_1 = WcsGeom.create(
        skydir=(266.405, -28.936),
        npix=(11, 11),
        binsz=0.1,
        axes=[axis1, axis2],
        frame="icrs",
    )
    geom_wcs_2 = WcsGeom.create(
        skydir=(0, 0), npix=(11, 11), binsz=0.1, frame="galactic"
    )

    spatial_data = np.zeros((11, 11))
    energy_data = axis1.center.reshape(3, 1, 1)
    time_data = axis2.center.reshape(4, 1, 1, 1)
    data = spatial_data + energy_data + 0.5 * time_data
    m = WcsNDMap(geom_wcs_1, data=data)

    m_r = m.reproject_to_geom(geom_wcs_2)
    assert m.data.shape == m_r.data.shape

    geom_wcs_3 = geom_wcs_1.copy(axes=[axis1, axis2])
    m_r_3 = m.reproject_to_geom(geom_wcs_3, preserve_counts=False)
    for data, idx in m_r_3.iter_by_image_data():
        ref = idx[1] + 0.5 * idx[0]
        if np.any(data > 0):
            assert_allclose(np.nanmean(data[data > 0]), ref)

    factor = 2
    axis1_up = axis1.upsample(factor=factor)
    axis2_up = axis2.upsample(factor=factor)
    geom_wcs_4 = geom_wcs_1.copy(axes=[axis1_up, axis2_up])
    m_r_4 = m.reproject_to_geom(geom_wcs_4, preserve_counts=False)
    for data, idx in m_r_4.iter_by_image_data():
        ref = int(idx[1] / factor) + 0.5 * int(idx[0] / factor)
        if np.any(data > 0):
            assert_allclose(np.nanmean(data[data > 0]), ref)


def test_wcsndmap_reproject_allsky_car():
    geom = WcsGeom.create(binsz=10.0, proj="CAR", frame="icrs")
    m = WcsNDMap(geom)
    mask = m.geom.get_coord()[0] > 180 * u.deg
    m.data = mask.astype(float)

    geom0 = WcsGeom.create(
        binsz=20, proj="CAR", frame="icrs", skydir=(180.0, 0.0), width=20
    )
    expected = np.mean([m.data[0, 0], m.data[0, -1]])  # wrap at 180deg

    m0 = m.reproject_to_geom(geom0)
    assert_allclose(m0.data[0], expected)

    geom1 = HpxGeom.create(binsz=5.0, frame="icrs")
    m1 = m.reproject_to_geom(geom1)
    mask = np.abs(m1.geom.get_coord()[0] - 180) <= 5
    assert_allclose(np.unique(m1.data[mask])[1], expected)


def test_iter_by_image():
    time_axis = MapAxis.from_bounds(
        0, 3, nbin=3, unit="hour", name="time", interp="lin"
    )

    energy_axis = MapAxis.from_bounds(
        1, 100, nbin=4, unit="TeV", name="energy", interp="log"
    )

    m_4d = Map.create(
        binsz=0.2, width=(1, 1), frame="galactic", axes=[energy_axis, time_axis]
    )

    for m in m_4d.iter_by_image(keepdims=True):
        assert m.data.ndim == 4
        assert m.geom.axes.names == ["energy", "time"]

    for m in m_4d.iter_by_image(keepdims=False):
        assert m.data.ndim == 2
        assert m.geom.axes.names == []


def test_iter_by_axis():
    time_axis = MapAxis.from_bounds(
        0, 3, nbin=3, unit="hour", name="time", interp="lin"
    )

    energy_axis = MapAxis.from_bounds(
        1, 100, nbin=4, unit="TeV", name="energy", interp="log"
    )

    m_4d = Map.create(
        binsz=0.2, width=(1, 1), frame="galactic", axes=[energy_axis, time_axis]
    )

    for m in m_4d.iter_by_axis(axis_name="energy", keepdims=False):
        assert m.data.ndim == 3
        assert m.geom.axes.names == ["time"]


def test_split_by_axis():
    time_axis = MapAxis.from_bounds(
        0,
        24,
        nbin=3,
        unit="hour",
        name="time",
    )
    energy_axis = MapAxis.from_bounds(
        1, 100, nbin=2, unit="TeV", name="energy", interp="log"
    )
    m_4d = Map.create(
        binsz=1.0, width=(10, 5), frame="galactic", axes=[energy_axis, time_axis]
    )
    maps = m_4d.split_by_axis("time")
    assert len(maps) == 3


def test_rename_axes():
    time_axis = MapAxis.from_bounds(
        0,
        24,
        nbin=3,
        unit="hour",
        name="time",
    )
    energy_axis = MapAxis.from_bounds(
        1, 100, nbin=2, unit="TeV", name="energy", interp="log"
    )
    m_4d = Map.create(
        binsz=1.0, width=(10, 5), frame="galactic", axes=[energy_axis, time_axis]
    )
    geom = m_4d.geom.rename_axes("energy", "energy_true")
    assert m_4d.geom.axes.names == ["energy", "time"]
    assert geom.axes.names == ["energy_true", "time"]

    new_map = m_4d.rename_axes("energy", "energy_true")
    assert m_4d.geom.axes.names == ["energy", "time"]
    assert new_map.geom.axes.names == ["energy_true", "time"]

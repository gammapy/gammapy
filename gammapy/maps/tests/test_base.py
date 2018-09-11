# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from collections import OrderedDict
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.units import Unit, Quantity
from ...utils.testing import requires_dependency
from ..base import Map
from ..geom import MapAxis
from ..wcs import WcsGeom
from ..wcsnd import WcsNDMap
from ..hpx import HpxGeom
from ..hpxnd import HpxNDMap

pytest.importorskip("scipy")
pytest.importorskip("healpy")
pytest.importorskip("numpy", "1.12.0")

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
    (0.1, 10.0, "hpx-sparse", SkyCoord(0.0, 30.0, unit="deg"), None, ""),
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

    m_copy = m.copy(meta=OrderedDict([("is_copy", True)]))
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
    assert_equal(m.geom.shape, sliced.geom.shape)

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


@pytest.mark.parametrize("map_type", ["wcs", "hpx", "hpx-sparse"])
def test_map_meta_read_write(map_type):
    meta = OrderedDict([("user", "test")])

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

    assert isinstance(m.geom, WcsGeom)
    m.geom = m.geom
    assert isinstance(m.geom, WcsGeom)

    assert isinstance(m.unit, u.CompositeUnit)
    assert m.unit == ""
    m.unit = "cm-2 s-1"
    assert m.unit.to_string() == "1 / (cm2 s)"

    assert isinstance(m.meta, OrderedDict)
    m.meta = {"spam": 42}
    assert isinstance(m.meta, OrderedDict)

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


@requires_dependency("reproject")
def test_map_reproject_wcs_to_wcs():
    energy_nodes = np.arange(3)
    time_nodes = np.arange(4)

    axis1 = MapAxis(energy_nodes, interp="lin", name="energy", node_type="center")
    axis2 = MapAxis(time_nodes, interp="lin", name="time", node_type="center")
    geom_wcs_1 = WcsGeom.create(
        skydir=(266.405, -28.936),
        npix=(11, 11),
        binsz=0.1,
        axes=[axis1, axis2],
        coordsys="CEL",
    )
    geom_wcs_2 = WcsGeom.create(skydir=(0, 0), npix=(11, 11), binsz=0.1, coordsys="GAL")

    spatial_data = np.zeros((11, 11))
    energy_data = energy_nodes.reshape(3, 1, 1)
    time_data = time_nodes.reshape(4, 1, 1, 1)
    data = spatial_data + energy_data + 0.5 * time_data
    m = WcsNDMap(geom_wcs_1, data=data)

    m_r = m.reproject(geom_wcs_2)

    assert m.data.shape == m_r.data.shape

    for data, idx in m_r.iter_by_image():
        ref = idx[1] + 0.5 * idx[0]
        assert_allclose(np.nanmean(data), ref)


@requires_dependency("reproject")
def test_map_reproject_wcs_to_hpx():
    axis = MapAxis.from_bounds(
        1.0, 10.0, 3, interp="log", name="energy", node_type="center"
    )
    geom_wcs = WcsGeom.create(
        skydir=(0, 0), npix=(11, 11), binsz=10, axes=[axis], coordsys="GAL"
    )
    geom_hpx = HpxGeom.create(binsz=10, coordsys="GAL", axes=[axis])

    data = np.arange(11 * 11 * 3).reshape(geom_wcs.data_shape)
    m = WcsNDMap(geom_wcs, data=data)

    m_r = m.reproject(geom_hpx)
    actual = m_r.get_by_coord({"lon": 0, "lat": 0, "energy": [1., 3.16227766, 10.]})
    assert_allclose(actual, [65., 186., 307.], rtol=1e-3)


@requires_dependency("reproject")
def test_map_reproject_hpx_to_wcs():
    axis = MapAxis.from_bounds(
        1.0, 10.0, 3, interp="log", name="energy", node_type="center"
    )
    geom_wcs = WcsGeom.create(
        skydir=(0, 0), npix=(11, 11), binsz=10, axes=[axis], coordsys="GAL"
    )
    geom_hpx = HpxGeom.create(binsz=10, coordsys="GAL", axes=[axis])

    data = np.arange(3 * 768).reshape(geom_hpx.data_shape)
    m = HpxNDMap(geom_hpx, data=data)

    m_r = m.reproject(geom_wcs)
    actual = m_r.get_by_coord({"lon": 0, "lat": 0, "energy": [1., 3.16227766, 10.]})
    assert_allclose(actual, [287.5, 1055.5, 1823.5], rtol=1e-3)

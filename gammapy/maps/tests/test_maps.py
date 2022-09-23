# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from gammapy.maps import Map, MapAxis, Maps, RegionNDMap, WcsGeom
from gammapy.utils.testing import assert_allclose


@pytest.fixture()
def map_dictionary():
    mapdict = {}
    axis = MapAxis.from_edges([1, 2, 3, 4], name="axis", unit="cm")
    geom = WcsGeom.create(npix=10, axes=[axis])
    mapdict["map1"] = Map.from_geom(geom, data=1)
    mapdict["map2"] = Map.from_geom(geom, data=2)
    return mapdict


def test_maps(map_dictionary):
    maps = Maps(**map_dictionary)

    maps["map3"] = maps["map1"].copy()

    assert maps.geom.npix[0] == 10
    assert len(maps) == 3
    assert_allclose(maps["map1"].data, 1)
    assert_allclose(maps["map2"].data, 2)
    assert_allclose(maps["map3"].data, 1)
    assert "map3" in maps.__str__()


@pytest.mark.xfail
def test_maps_wrong_addition(map_dictionary):
    maps = Maps(**map_dictionary)

    # Test pop method
    some_map = maps.pop("map2")
    assert len(maps) == 1
    assert_allclose(some_map.data, 2)

    # Test incorrect map addition
    with pytest.raises(ValueError):
        maps["map3"] = maps["map1"].sum_over_axes()


def test_maps_read_write(map_dictionary):
    maps = Maps(**map_dictionary)
    maps.write("test.fits", overwrite=True)
    new_maps = Maps.read("test.fits")

    assert new_maps.geom == maps.geom
    assert len(new_maps) == 2
    assert_allclose(new_maps["map1"].data, 1)
    assert_allclose(new_maps["map2"].data, 2)


def test_maps_region():
    axis = MapAxis.from_edges([1, 2, 3, 4], name="axis", unit="cm")
    map1 = RegionNDMap.create(region=None, axes=[axis])
    map1.data = 1
    map2 = RegionNDMap.create(region=None, axes=[axis])

    maps = Maps(map1=map1, map2=map2)

    assert len(maps) == 2
    assert_allclose(maps["map1"], 1)


def test_maps_from_geom():
    geom = WcsGeom.create(npix=5)
    names = ["map1", "map2", "map3"]
    kwargs_list = [
        {"unit": "cm2s", "dtype": "float64"},
        {"dtype": "bool"},
        {"data": np.arange(25).reshape(5, 5)},
    ]

    maps = Maps.from_geom(geom, names)
    maps_kwargs = Maps.from_geom(geom, names, kwargs_list=kwargs_list)

    assert len(maps) == 3
    assert maps["map1"].geom == geom
    assert maps["map2"].unit == ""
    assert maps["map3"].data.dtype == np.float32
    assert len(maps_kwargs) == 3
    assert maps_kwargs["map1"].unit == "cm2s"
    assert maps_kwargs["map1"].data.dtype == np.float64
    assert maps_kwargs["map2"].data.dtype == np.bool
    assert maps_kwargs["map3"].data[2, 2] == 12

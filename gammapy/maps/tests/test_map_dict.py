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


def test_map_dict(map_dictionary):
    map_dict = Maps(**map_dictionary)

    map_dict["map3"] = map_dict["map1"].copy()

    assert map_dict.geom.npix[0] == 10
    assert len(map_dict) == 3
    assert_allclose(map_dict["map1"].data, 1)
    assert_allclose(map_dict["map2"].data, 2)
    assert_allclose(map_dict["map3"].data, 1)
    assert "map3" in map_dict.__str__()


def test_map_dict_wrong_addition(map_dictionary):
    map_dict = Maps(**map_dictionary)

    # Test pop method
    some_map = map_dict.pop("map2")
    assert len(map_dict) == 1
    assert_allclose(some_map.data, 2)

    # Test incorrect map addition
    with pytest.raises(ValueError):
        map_dict["map3"] = map_dict["map1"].sum_over_axes()


def test_map_dict_read_write(map_dictionary):
    map_dict = Maps(**map_dictionary)
    map_dict.write("test.fits", overwrite=True)
    new_map_dict = Maps.read("test.fits")

    assert new_map_dict.geom == map_dict.geom
    assert len(new_map_dict) == 2
    assert_allclose(new_map_dict["map1"].data, 1)
    assert_allclose(new_map_dict["map2"].data, 2)


def test_map_dict_region():
    axis = MapAxis.from_edges([1, 2, 3, 4], name="axis", unit="cm")
    map1 = RegionNDMap.create(region=None, axes=[axis])
    map1.data = 1
    map2 = RegionNDMap.create(region=None, axes=[axis])

    map_dict = Maps(map1=map1, map2=map2)

    assert len(map_dict) == 2
    assert_allclose(map_dict["map1"], 1)

def test_map_dict_from_geom():
    geom = WcsGeom.create(npix=5)
    names = ["map1", "map2", "map3"]
    kwargs_list = [
        {"unit":"cm2s", "dtype":"float64"},
        {"dtype" : "bool"},
        {"data" : np.arange(25).reshape(5,5)}
    ]

    maps = Maps.from_geom(geom, names)
    maps_kwargs = Maps.from_geom(geom, names, kwargs_list=kwargs_list)

    assert len(maps)==3
    assert maps["map1"].geom == geom
    assert maps["map2"].unit == ''
    assert maps["map3"].data.dtype == np.float32
    assert len(maps_kwargs)==3
    assert maps_kwargs["map1"].unit == 'cm2s'
    assert maps_kwargs["map1"].data.dtype == np.float64
    assert maps_kwargs["map2"].data.dtype == np.bool
    assert maps_kwargs["map3"].data[2,2] == 12

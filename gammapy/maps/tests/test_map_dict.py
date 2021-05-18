# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from gammapy.maps import Map, MapAxis, MapDict, RegionNDMap, WcsGeom
from gammapy.utils.testing import assert_allclose

@pytest.fixture()
def map_dictionary():
    mapdict = {}
    axis = MapAxis.from_edges([1,2,3,4], name="axis", unit="cm")
    geom = WcsGeom.create(npix=10, axes=[axis])
    mapdict["map1"] = Map.from_geom(geom, data=1)
    mapdict["map2"] = Map.from_geom(geom, data=2)
    return mapdict

def test_map_dic(map_dictionary):
    map_dict = MapDict(**map_dictionary)

    map_dict["map3"] = map_dict["map1"].copy()

    assert map_dict.geom.npix[0] == 10
    assert len(map_dict) == 3
    assert_allclose(map_dict["map1"].data, 1)
    assert_allclose(map_dict["map2"].data, 2)
    assert_allclose(map_dict["map3"].data, 1)


def test_map_dict_read_write(map_dictionary):
    map_dict = MapDict(**map_dictionary)
    map_dict.write('test.fits', overwrite=True)
    new_map_dict = MapDict.read('test.fits')

    assert new_map_dict.geom == map_dict.geom
    assert len(new_map_dict) == 2
    assert_allclose(new_map_dict["MAP1"].data, 1)
    assert_allclose(new_map_dict["MAP2"].data, 2)


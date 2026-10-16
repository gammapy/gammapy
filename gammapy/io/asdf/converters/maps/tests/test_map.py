# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest

from gammapy.maps import MapAxis, WcsGeom, Map, Maps

asdf = pytest.importorskip("asdf")
pytest.importorskip("asdf.testing")


def test_maps_roundtrip(tmp_path):
    file_path = tmp_path / "test.asdf"
    maps = Maps()
    axis = MapAxis.from_edges([1, 2, 3, 4], name="axis", unit="cm")
    geom = WcsGeom.create(npix=10, axes=[axis])
    maps["map1"] = Map.from_geom(geom, data=1, meta={"testing": "map1"}, unit="cm")
    maps["map2"] = Map.from_geom(geom, data=2, meta={"testing": "map2"}, unit="m2")

    with asdf.AsdfFile() as af:
        af["maps"] = maps
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        result = af["maps"]
        assert result.keys() == maps.keys()
        for key in maps:
            assert maps[key].is_allclose(result[key])

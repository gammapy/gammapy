# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
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


def test_reproject_2d():
    npix1=3 
    geom1 = WcsGeom.create(npix=npix1, frame="icrs")
    geom1_large = WcsGeom.create(npix=npix1+5, frame="icrs")
    map1 = Map.from_geom(geom1, data=np.eye(npix1))

    factor = 10
    binsz = 0.5 / factor
    npix2 = 7 * factor 
    geom2 = WcsGeom.create(skydir=SkyCoord(0.1,0.1, unit=u.deg),
                           binsz=binsz,
                           npix=npix2,
                           frame="galactic"
                           )

    map1_repro = map1.reproject(geom2, preserve_counts=True)
    assert_allclose(np.sum(map1_repro), np.sum(map1), rtol=1e-5)
    map1_new = map1_repro.reproject(geom1_large, preserve_counts=True)
    assert_allclose(np.sum(map1_repro), np.sum(map1_new), rtol=1e-5)

    map1_repro = map1.reproject(geom2, preserve_counts=False)
    map1_new = map1_repro.reproject(geom1_large, preserve_counts=False)
    assert_allclose(np.sum(map1_repro*geom2.solid_angle()),
                    np.sum(map1_new*geom1_large.solid_angle()),
                    rtol=1e-3
                    )
    
    factor = 0.5
    binsz = 0.5 / factor
    npix = 7
    geom2 = WcsGeom.create(skydir=SkyCoord(0.1,0.1, unit=u.deg),
                           binsz=binsz,
                           npix=npix,
                           frame="galactic"
                           )
    geom1_large = WcsGeom.create(npix=npix1+5, frame="icrs")

    map1_repro = map1.reproject(geom2, preserve_counts=True)
    assert_allclose(np.sum(map1_repro), np.sum(map1), rtol=1e-5)
    map1_new = map1_repro.reproject(geom1_large, preserve_counts=True)
    assert_allclose(np.sum(map1_repro), np.sum(map1_new), rtol=1e-3)

    map1_repro = map1.reproject(geom2, preserve_counts=False)
    map1_new = map1_repro.reproject(geom1_large, preserve_counts=False)
    assert_allclose(np.sum(map1_repro*geom2.solid_angle()),
                    np.sum(map1_new*geom1_large.solid_angle()),
                    rtol=1e-3
                    )
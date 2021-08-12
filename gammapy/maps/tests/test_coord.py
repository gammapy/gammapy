# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import SkyCoord
from gammapy.maps import MapCoord


def test_mapcoord_repr():
    coord = MapCoord({"lon": 0, "lat": 0, "energy": 5})
    assert "MapCoord" in repr(coord)


def test_mapcoords_create():
    # From existing MapCoord
    coords_cel = MapCoord.create((0.0, 1.0), frame="icrs")
    coords_gal = MapCoord.create(coords_cel, frame="galactic")
    assert_allclose(coords_gal.lon, coords_cel.skycoord.galactic.l.deg)
    assert_allclose(coords_gal.lat, coords_cel.skycoord.galactic.b.deg)

    # 2D Tuple of scalars
    coords = MapCoord.create((0.0, 1.0))
    assert_allclose(coords.lon, 0.0)
    assert_allclose(coords.lat, 1.0)
    assert_allclose(coords[0], 0.0)
    assert_allclose(coords[1], 1.0)
    assert coords.frame is None
    assert coords.ndim == 2

    # 3D Tuple of scalars
    coords = MapCoord.create((0.0, 1.0, 2.0))
    assert_allclose(coords[0], 0.0)
    assert_allclose(coords[1], 1.0)
    assert_allclose(coords[2], 2.0)
    assert coords.frame is None
    assert coords.ndim == 3

    # 2D Tuple w/ NaN coordinates
    coords = MapCoord.create((np.nan, np.nan))

    # 2D Tuple w/ NaN coordinates
    lon, lat = np.array([np.nan, 1.0]), np.array([np.nan, 3.0])
    coords = MapCoord.create((lon, lat))
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)

    # 2D Tuple w/ SkyCoord
    lon, lat = np.array([0.0, 1.0]), np.array([2.0, 3.0])
    energy = np.array([100.0, 1000.0])
    skycoord_cel = SkyCoord(lon, lat, unit="deg", frame="icrs")
    skycoord_gal = SkyCoord(lon, lat, unit="deg", frame="galactic")
    coords = MapCoord.create((skycoord_cel,))
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert coords.frame == "icrs"
    assert coords.ndim == 2

    coords = MapCoord.create((skycoord_gal,))
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert coords.frame == "galactic"
    assert coords.ndim == 2

    # SkyCoord
    coords = MapCoord.create(skycoord_cel)
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert coords.frame == "icrs"
    assert coords.ndim == 2
    coords = MapCoord.create(skycoord_gal)
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert coords.frame == "galactic"
    assert coords.ndim == 2

    # 2D dict w/ vectors
    coords = MapCoord.create(dict(lon=lon, lat=lat))
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert coords.ndim == 2

    # 3D dict w/ vectors
    coords = MapCoord.create(dict(lon=lon, lat=lat, energy=energy))
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert_allclose(coords["energy"], energy)
    assert coords.ndim == 3

    # 3D dict w/ SkyCoord
    coords = MapCoord.create(dict(skycoord=skycoord_cel, energy=energy))
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert_allclose(coords["energy"], energy)
    assert coords.frame == "icrs"
    assert coords.ndim == 3

    # 3D dict  w/ vectors
    coords = MapCoord.create({"energy": energy, "lat": lat, "lon": lon})
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert_allclose(coords["energy"], energy)
    assert_allclose(coords[0], energy)
    assert_allclose(coords[1], lat)
    assert_allclose(coords[2], lon)
    assert coords.ndim == 3

    # Quantities
    coords = MapCoord.create(dict(energy=energy * u.TeV, lat=lat, lon=lon))
    assert coords["energy"].unit == "TeV"


def test_mapcoords_to_frame():
    lon, lat = np.array([0.0, 1.0]), np.array([2.0, 3.0])
    energy = np.array([100.0, 1000.0])
    skycoord_cel = SkyCoord(lon, lat, unit="deg", frame="icrs")
    skycoord_gal = SkyCoord(lon, lat, unit="deg", frame="galactic")

    coords = MapCoord.create(dict(lon=lon, lat=lat, energy=energy), frame="icrs")
    assert coords.frame == "icrs"
    assert_allclose(coords.skycoord.transform_to("icrs").ra.deg, skycoord_cel.ra.deg)
    assert_allclose(coords.skycoord.transform_to("icrs").dec.deg, skycoord_cel.dec.deg)
    coords = coords.to_frame("galactic")
    assert coords.frame == "galactic"
    assert_allclose(
        coords.skycoord.transform_to("galactic").l.deg, skycoord_cel.galactic.l.deg
    )
    assert_allclose(
        coords.skycoord.transform_to("galactic").b.deg, skycoord_cel.galactic.b.deg
    )

    coords = MapCoord.create(dict(lon=lon, lat=lat, energy=energy), frame="galactic")
    assert coords.frame == "galactic"
    assert_allclose(coords.skycoord.transform_to("galactic").l.deg, skycoord_gal.l.deg)
    assert_allclose(coords.skycoord.transform_to("galactic").b.deg, skycoord_gal.b.deg)
    coords = coords.to_frame("icrs")
    assert coords.frame == "icrs"
    assert_allclose(
        coords.skycoord.transform_to("icrs").ra.deg, skycoord_gal.icrs.ra.deg
    )
    assert_allclose(
        coords.skycoord.transform_to("icrs").dec.deg, skycoord_gal.icrs.dec.deg
    )

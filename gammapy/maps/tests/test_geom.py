# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from collections import OrderedDict
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import SkyCoord
from ..geom import MapAxis, MapCoord

pytest.importorskip("scipy")

mapaxis_geoms = [
    (np.array([0.25, 0.75, 1.0, 2.0]), "lin"),
    (np.array([0.25, 0.75, 1.0, 2.0]), "log"),
    (np.array([0.25, 0.75, 1.0, 2.0]), "sqrt"),
]

mapaxis_geoms_node_type = [
    (np.array([0.25, 0.75, 1.0, 2.0]), "lin", "edges"),
    (np.array([0.25, 0.75, 1.0, 2.0]), "log", "edges"),
    (np.array([0.25, 0.75, 1.0, 2.0]), "sqrt", "edges"),
    (np.array([0.25, 0.75, 1.0, 2.0]), "lin", "center"),
    (np.array([0.25, 0.75, 1.0, 2.0]), "log", "center"),
    (np.array([0.25, 0.75, 1.0, 2.0]), "sqrt", "center"),
]


@pytest.mark.parametrize(("edges", "interp"), mapaxis_geoms)
def test_mapaxis_init_from_edges(edges, interp):
    axis = MapAxis(edges, interp=interp)
    assert_allclose(axis.edges, edges)
    assert_allclose(axis.nbin, len(edges) - 1)


@pytest.mark.parametrize(("nodes", "interp"), mapaxis_geoms)
def test_mapaxis_from_nodes(nodes, interp):
    axis = MapAxis.from_nodes(nodes, interp=interp)
    assert_allclose(axis.center, nodes)
    assert_allclose(axis.nbin, len(nodes))


@pytest.mark.parametrize(("nodes", "interp"), mapaxis_geoms)
def test_mapaxis_from_bounds(nodes, interp):
    axis = MapAxis.from_bounds(nodes[0], nodes[-1], 3, interp=interp)
    assert_allclose(axis.edges[0], nodes[0])
    assert_allclose(axis.edges[-1], nodes[-1])
    assert_allclose(axis.nbin, 3)


@pytest.mark.parametrize(("nodes", "interp", "node_type"), mapaxis_geoms_node_type)
def test_mapaxis_pix_to_coord(nodes, interp, node_type):
    axis = MapAxis(nodes, interp=interp, node_type=node_type)
    assert_allclose(axis.center, axis.pix_to_coord(np.arange(axis.nbin, dtype=float)))
    assert_allclose(
        np.arange(axis.nbin + 1, dtype=float) - 0.5, axis.coord_to_pix(axis.edges)
    )


@pytest.mark.parametrize(("nodes", "interp", "node_type"), mapaxis_geoms_node_type)
def test_mapaxis_coord_to_idx(nodes, interp, node_type):
    axis = MapAxis(nodes, interp=interp, node_type=node_type)
    assert_allclose(np.arange(axis.nbin, dtype=int), axis.coord_to_idx(axis.center))


@pytest.mark.parametrize(("nodes", "interp", "node_type"), mapaxis_geoms_node_type)
def test_mapaxis_slice(nodes, interp, node_type):
    axis = MapAxis(nodes, interp=interp, node_type=node_type)
    saxis = axis.slice(slice(1, 3))
    assert_allclose(saxis.nbin, 2)
    assert_allclose(saxis.center, axis.center[slice(1, 3)])

    axis = MapAxis(nodes, interp=interp, node_type=node_type)
    saxis = axis.slice(slice(1, None))
    assert_allclose(saxis.nbin, axis.nbin - 1)
    assert_allclose(saxis.center, axis.center[slice(1, None)])

    axis = MapAxis(nodes, interp=interp, node_type=node_type)
    saxis = axis.slice(slice(None, 2))
    assert_allclose(saxis.nbin, 2)
    assert_allclose(saxis.center, axis.center[slice(None, 2)])

    axis = MapAxis(nodes, interp=interp, node_type=node_type)
    saxis = axis.slice(slice(None, -1))
    assert_allclose(saxis.nbin, axis.nbin - 1)
    assert_allclose(saxis.center, axis.center[slice(None, -1)])


def test_mapcoords_create():
    # From existing MapCoord
    coords_cel = MapCoord.create((0.0, 1.0), coordsys="CEL")
    coords_gal = MapCoord.create(coords_cel, coordsys="GAL")
    assert_allclose(coords_gal.lon, coords_cel.skycoord.galactic.l.deg)
    assert_allclose(coords_gal.lat, coords_cel.skycoord.galactic.b.deg)

    # 2D Tuple of scalars
    coords = MapCoord.create((0.0, 1.0))
    assert_allclose(coords.lon, 0.0)
    assert_allclose(coords.lat, 1.0)
    assert_allclose(coords[0], 0.0)
    assert_allclose(coords[1], 1.0)
    assert coords.coordsys is None
    assert coords.ndim == 2

    # 3D Tuple of scalars
    coords = MapCoord.create((0.0, 1.0, 2.0))
    assert_allclose(coords[0], 0.0)
    assert_allclose(coords[1], 1.0)
    assert_allclose(coords[2], 2.0)
    assert coords.coordsys is None
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
    energy = np.array([100., 1000.])
    skycoord_cel = SkyCoord(lon, lat, unit="deg", frame="icrs")
    skycoord_gal = SkyCoord(lon, lat, unit="deg", frame="galactic")
    coords = MapCoord.create((skycoord_cel,))
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert coords.coordsys == "CEL"
    assert coords.ndim == 2
    coords = MapCoord.create((skycoord_gal,))
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert coords.coordsys == "GAL"
    assert coords.ndim == 2

    # SkyCoord
    coords = MapCoord.create(skycoord_cel)
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert coords.coordsys == "CEL"
    assert coords.ndim == 2
    coords = MapCoord.create(skycoord_gal)
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert coords.coordsys == "GAL"
    assert coords.ndim == 2

    # 2D Dict w/ vectors
    coords = MapCoord.create(dict(lon=lon, lat=lat))
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert coords.ndim == 2

    # 3D Dict w/ vectors
    coords = MapCoord.create(dict(lon=lon, lat=lat, energy=energy))
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert_allclose(coords["energy"], energy)
    assert coords.ndim == 3

    # 3D Dict w/ SkyCoord
    coords = MapCoord.create(dict(skycoord=skycoord_cel, energy=energy))
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)
    assert_allclose(coords["energy"], energy)
    assert coords.ndim == 3

    # 3D OrderedDict w/ vectors
    coords = MapCoord.create(
        OrderedDict([("energy", energy), ("lat", lat), ("lon", lon)])
    )
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


def test_mapcoords_to_coordsys():
    lon, lat = np.array([0.0, 1.0]), np.array([2.0, 3.0])
    energy = np.array([100., 1000.])
    skycoord_cel = SkyCoord(lon, lat, unit="deg", frame="icrs")
    skycoord_gal = SkyCoord(lon, lat, unit="deg", frame="galactic")

    coords = MapCoord.create(dict(lon=lon, lat=lat, energy=energy), coordsys="CEL")
    assert coords.coordsys == "CEL"
    assert_allclose(coords.skycoord.transform_to("icrs").ra.deg, skycoord_cel.ra.deg)
    assert_allclose(coords.skycoord.transform_to("icrs").dec.deg, skycoord_cel.dec.deg)
    coords = coords.to_coordsys("GAL")
    assert coords.coordsys == "GAL"
    assert_allclose(
        coords.skycoord.transform_to("galactic").l.deg, skycoord_cel.galactic.l.deg
    )
    assert_allclose(
        coords.skycoord.transform_to("galactic").b.deg, skycoord_cel.galactic.b.deg
    )

    coords = MapCoord.create(dict(lon=lon, lat=lat, energy=energy), coordsys="GAL")
    assert coords.coordsys == "GAL"
    assert_allclose(coords.skycoord.transform_to("galactic").l.deg, skycoord_gal.l.deg)
    assert_allclose(coords.skycoord.transform_to("galactic").b.deg, skycoord_gal.b.deg)
    coords = coords.to_coordsys("CEL")
    assert coords.coordsys == "CEL"
    assert_allclose(
        coords.skycoord.transform_to("icrs").ra.deg, skycoord_gal.icrs.ra.deg
    )
    assert_allclose(
        coords.skycoord.transform_to("icrs").dec.deg, skycoord_gal.icrs.dec.deg
    )


def test_mapaxis_repr():
    axis = MapAxis([1, 2, 3], name="test")
    assert "MapAxis" in repr(axis)


def test_mapcoord_repr():
    coord = MapCoord({"lon": 0, "lat": 0, "energy": 5})
    assert "MapCoord" in repr(coord)

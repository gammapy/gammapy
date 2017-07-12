# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from ..geom import MapAxis, MapCoords

pytest.importorskip('scipy')

mapaxis_geoms = [
    (np.array([0.25, 0.75, 1.0, 2.0]), 'lin'),
    (np.array([0.25, 0.75, 1.0, 2.0]), 'log'),
    (np.array([0.25, 0.75, 1.0, 2.0]), 'sqrt'),
]

mapaxis_geoms_node_type = [
    (np.array([0.25, 0.75, 1.0, 2.0]), 'lin', 'edge'),
    (np.array([0.25, 0.75, 1.0, 2.0]), 'log', 'edge'),
    (np.array([0.25, 0.75, 1.0, 2.0]), 'sqrt', 'edge'),
    (np.array([0.25, 0.75, 1.0, 2.0]), 'lin', 'center'),
    (np.array([0.25, 0.75, 1.0, 2.0]), 'log', 'center'),
    (np.array([0.25, 0.75, 1.0, 2.0]), 'sqrt', 'center'),
]


@pytest.mark.parametrize(('edges', 'interp'),
                         mapaxis_geoms)
def test_mapaxis_init_from_edges(edges, interp):
    axis = MapAxis(edges, interp=interp)
    assert_allclose(axis.edges, edges)
    assert_allclose(axis.nbin, len(edges) - 1)


@pytest.mark.parametrize(('nodes', 'interp'),
                         mapaxis_geoms)
def test_mapaxis_from_nodes(nodes, interp):
    axis = MapAxis.from_nodes(nodes, interp=interp)
    assert_allclose(axis.center, nodes)
    assert_allclose(axis.nbin, len(nodes))


@pytest.mark.parametrize(('nodes', 'interp'),
                         mapaxis_geoms)
def test_mapaxis_from_bounds(nodes, interp):
    axis = MapAxis.from_bounds(nodes[0], nodes[-1], 3,
                               interp=interp)
    assert_allclose(axis.edges[0], nodes[0])
    assert_allclose(axis.edges[-1], nodes[-1])
    assert_allclose(axis.nbin, 3)


@pytest.mark.parametrize(('nodes', 'interp', 'node_type'),
                         mapaxis_geoms_node_type)
def test_mapaxis_pix_to_coord(nodes, interp, node_type):
    axis = MapAxis(nodes, interp=interp, node_type=node_type)
    assert_allclose(axis.center,
                    axis.pix_to_coord(np.arange(axis.nbin, dtype=float)))
    assert_allclose(np.arange(axis.nbin + 1, dtype=float) - 0.5,
                    axis.coord_to_pix(axis.edges))


@pytest.mark.parametrize(('nodes', 'interp', 'node_type'),
                         mapaxis_geoms_node_type)
def test_mapaxis_coord_to_idx(nodes, interp, node_type):
    axis = MapAxis(nodes, interp=interp, node_type=node_type)
    assert_allclose(np.arange(axis.nbin, dtype=int),
                    axis.coord_to_idx(axis.center))


@pytest.mark.parametrize(('nodes', 'interp', 'node_type'),
                         mapaxis_geoms_node_type)
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
    # 2D Scalar
    coords = MapCoords.create((0.0, 0.0))

    # 2D Scalar w/ NaN coordinates
    coords = MapCoords.create((np.nan, np.nan))

    # 2D Vector w/ NaN coordinates
    lon, lat = np.array([np.nan, 1.0]), np.array([np.nan, 3.0])
    coords = MapCoords.create((lon, lat))
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)

    # 2D Vector w/ SkyCoord
    lon, lat = np.array([0.0, 1.0]), np.array([2.0, 3.0])
    coords = MapCoords.create((SkyCoord(lon, lat, unit='deg')))
    assert_allclose(coords.lon, lon)
    assert_allclose(coords.lat, lat)

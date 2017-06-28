# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from ..geom import MapAxis, MapCoords


mapaxis_test_geoms = [
    (np.array([0.25, 0.75, 1.0]), 'lin'),
    (np.array([0.25, 0.75, 1.0]), 'log'),
    (np.array([0.25, 0.75, 1.0]), 'sqrt')]


@pytest.mark.parametrize(('edges', 'interp'),
                         mapaxis_test_geoms)
def test_mapaxis_init_from_edges(edges, interp):

    axis = MapAxis(edges, interp=interp)
    assert_allclose(axis.edges, edges)
    assert_allclose(axis.nbin, len(edges) - 1)


@pytest.mark.parametrize(('nodes', 'interp'),
                         mapaxis_test_geoms)
def test_mapaxis_from_nodes(nodes, interp):

    axis = MapAxis.from_nodes(nodes, interp=interp)
    assert_allclose(axis.center, nodes)
    assert_allclose(axis.nbin, len(nodes))


@pytest.mark.parametrize(('nodes', 'interp'),
                         mapaxis_test_geoms)
def test_mapaxis_pix_to_coord(nodes, interp):

    axis = MapAxis.from_nodes(nodes, interp=interp)
    assert_allclose(axis.center,
                    axis.pix_to_coord(np.arange(axis.nbin, dtype=float)))
    assert_allclose(np.arange(axis.nbin + 1, dtype=float) - 0.5,
                    axis.coord_to_pix(axis.edges))


@pytest.mark.parametrize(('nodes', 'interp'),
                         mapaxis_test_geoms)
def test_mapaxis_coord_to_idx(nodes, interp):

    axis = MapAxis.from_nodes(nodes, interp=interp)
    assert_allclose(np.arange(axis.nbin, dtype=int),
                    axis.coord_to_idx(axis.center))

    axis = MapAxis.from_edges(nodes, interp=interp)
    assert_allclose(np.arange(axis.nbin, dtype=int),
                    axis.coord_to_idx(axis.center))


def test_mapcoords_create():

    # 2D Scalar
    coords = MapCoords.create((0.0,0.0))

    # 2D Scalar w/ NaN coordinates
    coords = MapCoords.create((np.nan,np.nan))

    # 2D Vector w/ NaN coordinates
    lon, lat = np.array([np.nan,1.0]), np.array([np.nan,3.0])
    coords = MapCoords.create((lon,lat))
    assert_allclose(coords.lon,lon)
    assert_allclose(coords.lat,lat)

    # 2D Vector w/ SkyCoord
    lon, lat = np.array([0.0,1.0]), np.array([2.0,3.0])
    coords = MapCoords.create((SkyCoord(lon,lat,unit='deg')))
    assert_allclose(coords.lon,lon)
    assert_allclose(coords.lat,lat)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.table import Table
from ...utils.testing import requires_dependency
from ...maps import MapAxis, HpxGeom, Map, WcsNDMap
from ...data import EventList
from ..counts import fill_map_counts

pytest.importorskip("scipy")


@pytest.fixture()
def events():
    t = Table()
    t["EVENT_ID"] = np.array([1, 5], dtype=np.uint16)
    t["RA"] = [5, 11] * u.deg
    t["DEC"] = [0, 0] * u.deg
    t["ENERGY"] = [10, 12] * u.TeV
    t["TIME"] = [3, 4] * u.s
    return EventList(t)


def test_fill_map_counts_wcs(events):
    # 2D map
    m = Map.create(npix=(2, 1), binsz=10)
    fill_map_counts(m, events)
    assert_allclose(m.data, [[1, 0]])

    # 3D with energy axis
    axis = MapAxis.from_edges([9, 11, 13], name="energy", unit="TeV")
    m = Map.create(npix=(2, 1), binsz=10, axes=[axis])
    fill_map_counts(m, events)
    assert m.data.sum() == 1
    assert_allclose(m.data[0, 0, 0], 1)


@requires_dependency("healpy")
def test_fill_map_counts_hpx(events):
    # 2D map
    m = Map.from_geom(HpxGeom(1))
    fill_map_counts(m, events)
    assert m.data[4] == 2

    # 3D with energy axis
    axis = MapAxis.from_edges([9, 11, 13], name="energy", unit="TeV")
    m = Map.from_geom(HpxGeom(1, axes=[axis]))
    fill_map_counts(m, events)
    assert m.data[0, 4] == 1
    assert m.data[1, 4] == 1


def test_fill_map_counts_keyerror(events):
    axis = MapAxis([0, 1, 2], name="nokey")
    m = WcsNDMap.create(binsz=0.1, npix=10, axes=[axis])
    with pytest.raises(KeyError):
        fill_map_counts(m, events)


def test_fill_map_counts_uint_column(events):
    # Check that unsigned int column works.
    # Regression test for https://github.com/gammapy/gammapy/issues/1620
    axis = MapAxis.from_edges([0, 3, 6], name="event_id")
    m = Map.create(npix=(2, 1), binsz=10, axes=[axis])
    fill_map_counts(m, events)
    assert m.data.sum() == 1
    assert_allclose(m.data[0, 0, 0], 1)

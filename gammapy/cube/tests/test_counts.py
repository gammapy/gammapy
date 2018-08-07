# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.table import Column
from astropy.coordinates import SkyCoord
import astropy.units as u
from ...utils.testing import requires_dependency, requires_data
from ...maps import MapAxis, WcsGeom, HpxGeom, Map, WcsNDMap
from ...data import EventList
from ..counts import fill_map_counts

pytest.importorskip('scipy')

# TODO: move these two cases to different test functions?
# Would allow to change the asserts in `test_fill_map_counts` to something much more specific / useful, no?
axis_energy_reco = MapAxis(np.logspace(-1., 1.5, 10), interp='log', node_type='edge',
                           name='energy_reco', unit='TeV')

# initial time of run 110380 from cta 1DC
times = np.linspace(664504199, 664504199 + 1900., 10)
axis_time = MapAxis(times, node_type='edge', name='time', unit='s')

pos_cta = SkyCoord(0.0, 0.0, frame='galactic', unit='deg')

geom_cta = {'binsz': 0.02, 'coordsys': 'GAL', 'width': 15 * u.deg,
            'skydir': pos_cta, 'axes': [axis_energy_reco]}
geom_cta_time = {'binsz': 0.02, 'coordsys': 'GAL', 'width': 15 * u.deg,
                 'skydir': pos_cta, 'axes': [axis_energy_reco, axis_time]}


# TODO: change the test event list to something that's created from scratch,
# using values so that it's possible to make simple assert statements on the
# map data in the tests below, i.e. have pixels that should receive 0, 1 or 2 counts
@pytest.fixture(scope='session')
def events():
    return EventList.read('$GAMMAPY_EXTRA/datasets/cta-1dc/data/baseline/gps/gps_baseline_110380.fits')


@requires_data('gammapy-extra')
@pytest.mark.parametrize('geom_opts', [geom_cta, geom_cta_time])
def test_fill_map_counts(geom_opts, events):
    geom = WcsGeom.create(**geom_opts)
    cntmap = Map.from_geom(geom)

    fill_map_counts(cntmap, events)

    # Compute expected number of entries in the map
    valid = np.ones_like(events.energy.value)
    for axis in geom.axes:
        if axis.name == 'energy_reco':
            valid = np.logical_and(events.energy.value >= axis.edges[0], valid)
            valid = np.logical_and(events.energy.value <= axis.edges[-1], valid)
        else:
            valid = np.logical_and(events.table[axis.name.upper()] >= axis.edges[0], valid)
            valid = np.logical_and(events.table[axis.name.upper()] <= axis.edges[-1], valid)

    assert cntmap.data.sum() == valid.sum()


@requires_data('gammapy-extra')
@requires_dependency('healpy')
def test_fill_map_counts_hpx(events):
    # This tests healpix maps fill with non standard non spatial axis

    axis_det = MapAxis([-2, 1, 5], node_type='edge', name='detx', unit='deg')
    # This test to check entries without units in eventlist table do not fail
    axis_evt = MapAxis((0, 100000, 150000), node_type='edge', name='event_id')

    geom = HpxGeom(256, coordsys='GAL', axes=[axis_evt, axis_det])

    m = Map.from_geom(geom)

    fill_map_counts(m, events)

    assert m.data[0].sum() == 66697
    assert m.data[1].sum() == 29410


@requires_data('gammapy-extra')
def test_fill_map_counts_keyerror(events):
    axis = MapAxis([0, 1, 2], node_type='edge', name='nokey')
    cntmap = WcsNDMap.create(binsz=0.1, npix=10, axes=[axis])
    with pytest.raises(KeyError):
        fill_map_counts(cntmap, events)

@requires_data('gammapy-extra')
def test_fill_map_counts_multiple_energy_axes(events):
    axis_mc = MapAxis([0, 2, 4], name='energy_mc', unit='TeV')
    axis_reco = MapAxis([0.001, 1000.], name='energy_reco', unit='TeV')
    axis = MapAxis([0.001, 1000.], name='energy', unit='TeV')

    cntmap1 = WcsNDMap.create(binsz=1, npix=10, coordsys='GAL', axes=[axis, axis_mc])
    cntmap2 = WcsNDMap.create(binsz=1, npix=10, coordsys='GAL', axes=[axis_mc, axis_reco])

    events.table['ENERGY_MC'] = 1*u.TeV

    # Check that energy_mc entries are placed in the right axis
    fill_map_counts(cntmap1, events)
    assert_allclose(cntmap1.data.sum(axis=(1, 2, 3)), [105592, 0])
    fill_map_counts(cntmap2, events)
    assert_allclose(cntmap2.data.sum(axis=(0, 2, 3)), [105592, 0])

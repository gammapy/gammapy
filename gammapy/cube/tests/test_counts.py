# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from ...utils.testing import requires_dependency, requires_data
from ...maps import MapAxis, WcsGeom, HpxGeom, Map
from ...data import DataStore, EventList
from ..new import fill_map_counts

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
def make_test_event_list():
    ds = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/cta-1dc/index/gps/')
    return ds.obs(110380).events


@pytest.fixture(scope='session')
def evt_2fhl():
    return EventList.read('$GAMMAPY_EXTRA/datasets/fermi_2fhl/2fhl_events.fits.gz')


@requires_data('gammapy-extra')
@pytest.mark.parametrize('geom_opts', [geom_cta, geom_cta_time])
def test_fill_map_counts(geom_opts):
    events = make_test_event_list()
    geom = WcsGeom.create(**geom_opts)
    cntmap = Map.from_geom(geom)

    fill_map_counts(cntmap, events)

    # Number of entries in the map
    nmap = cntmap.data.sum()
    # number of events
    valid = np.ones_like(events.energy.value)
    for axis in geom.axes:
        if axis.name == 'energy_reco':
            valid = np.logical_and(events.energy.value >= axis.edges[0], valid)
            valid = np.logical_and(events.energy.value <= axis.edges[-1], valid)
        else:
            valid = np.logical_and(events.table[axis.name.upper()] >= axis.edges[0], valid)
            valid = np.logical_and(events.table[axis.name.upper()] <= axis.edges[-1], valid)

    nevt = valid.sum()
    assert nmap == nevt


@requires_data('gammapy-extra')
@requires_dependency('healpy')
def test_fill_map_counts_hpx(evt_2fhl):
    # This tests healpix maps fill with non standard non spatial axis

    axis_zen = MapAxis([0, 45, 180], node_type='edge', name='zenith_angle', unit=u.deg)
    geom = HpxGeom(256, coordsys='GAL', axes=[axis_zen])
    m = Map.from_geom(geom)

    fill_map_counts(m, evt_2fhl)

    nmap_l = np.sum(m.data[0])
    nmap_h = np.sum(m.data[1])

    nevt_l = np.sum(evt_2fhl.table['ZENITH_ANGLE'] < 45)
    nevt_h = np.sum(evt_2fhl.table['ZENITH_ANGLE'] > 45)

    assert nmap_l == nevt_l
    assert nmap_h == nevt_h

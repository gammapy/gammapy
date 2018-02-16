# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from ...maps import MapAxis, WcsGeom, WcsNDMap
from ...data import DataStore
from ...cube.basic_cube import make_map_counts

# initial time of run 110380 from cta 1DC
t0_1DC = 664504199. * u.s

energy_reco = np.logspace(-1., 1.5, 10) * u.TeV
times = np.linspace(t0_1DC.value, t0_1DC.value+1900., 10)*u.s

axis_energy_reco = MapAxis(energy_reco.value, interp='log', node_type='edge',
                           name='energy_reco', unit = energy_reco.unit)
axis_time = MapAxis(times, node_type='edge', name='time',unit = times.unit)

cta_1dc_store = "$GAMMAPY_EXTRA/datasets/cta-1dc/index/gps/"
cta_1dc_runs = [110380]

pos_cta = SkyCoord(0.0, 0.0, frame='galactic',unit='deg')

geom_cta = {'binsz':0.02, 'coordsys':'GAL', 'width' : 15*u.deg,
            'skydir' : pos_cta, 'axes' : [axis_energy_reco]}
geom_cta_time = {'binsz':0.02, 'coordsys':'GAL', 'width' : 15*u.deg,
                 'skydir' : pos_cta, 'axes' : [axis_energy_reco, axis_time]}

@pytest.mark.parametrize("ds_path,run_list,geom",[(cta_1dc_store,cta_1dc_runs, geom_cta),
                                                  (cta_1dc_store,cta_1dc_runs, geom_cta_time)])
def test_counts_map_maker(ds_path, run_list, geom):
    # TODO: change the test event list to something that's created from scratch,
    # using values so that it's possible to make simple assert statements on the
    # map data in the tests below, i.e. have pixels that should receive 0, 1 or 2 counts

    # Get datastore
    ds = DataStore.from_dir(ds_path)
    run = run_list[0]

    # Get observation
    obs = ds.obs(run)
    events = obs.events

    # Build WcsGeom
    wcsgeom = WcsGeom.create(**geom)

    # Extract count map
    cntmap = make_map_counts(events, wcsgeom)

    # Number of entries in the map
    nmap = cntmap.data.sum()
    # number of events
    valid = np.ones_like(events.energy.value)
    for axis in geom['axes']:
        if axis.name == 'energy_reco':
            valid = np.logical_and(events.energy.value >= axis.edges[0], valid)
            valid = np.logical_and(events.energy.value <= axis.edges[-1], valid)
        else:
            valid = np.logical_and(events.table[axis.name.upper()]>=axis.edges[0],valid)
            valid = np.logical_and(events.table[axis.name.upper()]<=axis.edges[-1],valid)

    nevt = valid.sum()
    assert nmap == nevt

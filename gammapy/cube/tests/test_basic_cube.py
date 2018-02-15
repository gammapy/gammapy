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

energy_true = np.logspace(-2., 2., 30) * u.TeV
energy_reco = np.logspace(-1., 1.5, 10) * u.TeV
times = np.linspace(t0_1DC.value, t0_1DC.value+1900., 10)*u.s

axis_energy_true = MapAxis(energy_true.value, interp='log', node_type='edge',
                           name='energy_true', unit = energy_true.unit)
axis_energy_reco = MapAxis(energy_reco.value, interp='log', node_type='edge',
                           name='energy_reco', unit = energy_reco.unit)
axis_time = MapAxis(times, node_type='edge', name='time',unit = times.unit)

cta_1dc_store = "$GAMMAPY_EXTRA/datasets/cta-1dc/index/gps/"
cta_1dc_runs = [110380]

hess_crab_store = "$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/"
hess_crab_runs = [23523]

pos_crab = SkyCoord(83.633212, 22.01446, frame='icrs',unit='deg').galactic
pos_cta = SkyCoord(0.0, 0.0, frame='galactic',unit='deg')

geom_cta = {'binsz':0.02, 'coordsys':'GAL', 'width' : 15*u.deg,
            'skydir' : pos_cta, 'axes' : [axis_energy_reco]}
geom_cta_time = {'binsz':0.02, 'coordsys':'GAL', 'width' : 15*u.deg,
                 'skydir' : pos_cta, 'axes' : [axis_energy_reco, axis_time]}

geom_hess_true = {'binsz':0.02, 'coordsys':'GAL', 'width' : 10*u.deg,
                  'skydir' : pos_crab, 'axes' : [axis_energy_true]}

geom_hess_reco = {'binsz':0.02, 'coordsys':'GAL', 'width' : 10*u.deg,
                  'skydir' : pos_crab, 'axes' : [axis_energy_reco]}

@pytest.mark.parametrize("ds_path,run_list,geom",[(cta_1dc_store,cta_1dc_runs, geom_cta),
                                                  (cta_1dc_store,cta_1dc_runs, geom_cta_time),
                                                  (hess_crab_store,hess_crab_runs,geom_hess_reco)])
def test_counts_map_maker(ds_path, run_list, geom):
    # Get datastore
    ds = DataStore.from_dir(ds_path)
    run = run_list[0]

    # Get observation
    obs = ds.obs(run)
    evts = obs.events

    # Build WcsGeom
    wcsgeom = WcsGeom.create(**geom)
    # Build mask WcsNDMap
    mask = WcsNDMap(wcsgeom)
    # Fill it with ones
    mask.data += 1.

    # Extract count map
    cntmap = make_map_counts(evts, mask)

    # Number of entries in the map
    nmap = cntmap.data.sum()
    # number of events
    valid = np.ones_like(evts.energy.value)
    for axis in geom['axes']:
        if axis.name == 'energy_reco':
            valid = np.logical_and(evts.energy.value >= axis.edges[0], valid)
            valid = np.logical_and(evts.energy.value <= axis.edges[-1], valid)
        else:
            valid = np.logical_and(evts.table[axis.name.upper()]>=axis.edges[0],valid)
            valid = np.logical_and(evts.table[axis.name.upper()]<=axis.edges[-1],valid)

    nevt = valid.sum()
    print(nevt,nmap)
    assert nmap == nevt

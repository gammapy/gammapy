# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from ...maps import MapAxis, WcsGeom, HpxGeom, Map
from ...data import DataStore, EventList
from ...cube.basic_cube import fill_map_counts

# initial time of run 110380 from cta 1DC
t0_1DC = 664504199. * u.s

energy_reco = np.logspace(-1., 1.5, 10) * u.TeV
times = np.linspace(t0_1DC.value, t0_1DC.value + 1900., 10) * u.s

axis_energy_reco = MapAxis(energy_reco.value, interp='log', node_type='edge',
                           name='energy_reco', unit=energy_reco.unit)
axis_time = MapAxis(times, node_type='edge', name='time', unit=times.unit)

cta_1dc_store = "$GAMMAPY_EXTRA/datasets/cta-1dc/index/gps/"
cta_1dc_runs = [110380]

pos_cta = SkyCoord(0.0, 0.0, frame='galactic', unit='deg')

geom_cta = {'binsz': 0.02, 'coordsys': 'GAL', 'width': 15 * u.deg,
            'skydir': pos_cta, 'axes': [axis_energy_reco]}
geom_cta_time = {'binsz': 0.02, 'coordsys': 'GAL', 'width': 15 * u.deg,
                 'skydir': pos_cta, 'axes': [axis_energy_reco, axis_time]}


@pytest.mark.parametrize("ds_path,run_list,geom", [(cta_1dc_store, cta_1dc_runs, geom_cta),
                                                   (cta_1dc_store, cta_1dc_runs, geom_cta_time)])
def test_fill_map_counts(ds_path, run_list, geom):
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
    cntmap = Map.from_geom(wcsgeom)
    fill_map_counts(cntmap,events)

    # Number of entries in the map
    nmap = cntmap.data.sum()
    # number of events
    valid = np.ones_like(events.energy.value)
    for axis in geom['axes']:
        if axis.name == 'energy_reco':
            valid = np.logical_and(events.energy.value >= axis.edges[0], valid)
            valid = np.logical_and(events.energy.value <= axis.edges[-1], valid)
        else:
            valid = np.logical_and(events.table[axis.name.upper()] >= axis.edges[0], valid)
            valid = np.logical_and(events.table[axis.name.upper()] <= axis.edges[-1], valid)

    nevt = valid.sum()
    assert nmap == nevt


def test_fill_map_counts_fermi():
    # This tests healpix maps fill with non standard non spatial axis
    angles = np.array((0,45,180))*u.deg
    axis_zen = MapAxis(angles, node_type='edge', name='zenith_angle', unit=u.deg)

    evt_2fhl = EventList.read('$GAMMAPY_EXTRA/datasets/fermi_2fhl/2fhl_events.fits.gz')
    hpxgeom = HpxGeom(256,coordsys='GAL',axes=[axis_zen])
    map_2fhl = Map.from_geom(hpxgeom)
    fill_map_counts(map_2fhl,evt_2fhl)

    nmap_l = np.sum(map_2fhl.data[0])
    nmap_h = np.sum(map_2fhl.data[1])

    nevt_l = np.sum(evt_2fhl.table['ZENITH_ANGLE']<45)
    nevt_h = np.sum(evt_2fhl.table['ZENITH_ANGLE']>45)

    assert nmap_l == nevt_l
    assert nmap_h == nevt_h

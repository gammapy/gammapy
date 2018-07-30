# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
import pytest
from ...data import DataStore
from ...utils.testing import requires_data, requires_dependency


@pytest.fixture(scope='session')
def data_store():
    return DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_datastore_hd_hap(data_store):
    """Test HESS HAP-HD data access."""
    obs = data_store.obs(obs_id=23523)

    assert str(type((obs.events))) == "<class 'gammapy.data.event_list.EventList'>"
    assert str(type(obs.gti)) == "<class 'gammapy.data.gti.GTI'>"
    assert str(type(obs.aeff)) == "<class 'gammapy.irf.effective_area.EffectiveAreaTable2D'>"
    assert str(type(obs.edisp)) == "<class 'gammapy.irf.energy_dispersion.EnergyDispersion2D'>"
    assert str(type(obs.psf)) == "<class 'gammapy.irf.psf_analytical.EnergyDependentMultiGaussPSF'>"
    # TODO: no background model available yet
    # assert str(type(obs.bkg)) == ""


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_datastore_pa():
    """Test HESS ParisAnalysis data access."""
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-pa')

    obs = data_store.obs(obs_id=23523)
    filename = str(obs.location(hdu_type='bkg').path(abs_path=False))
    assert filename == 'background/bgmodel_alt7_az0.fits.gz'

    # assert str(type((obs.events))) == "<class 'gammapy.data.event_list.EventList'>"
    # TODO: GTI is not listed in PA HDU index table.
    # For now maybe add a workaround to find it in the same file as the events HDU?
    # assert str(type(obs.gti)) == "<class 'gammapy.data.gti.GTI'>"
    assert str(type(obs.aeff)) == "<class 'gammapy.irf.effective_area.EffectiveAreaTable2D'>"
    assert str(type(obs.edisp)) == "<class 'gammapy.irf.energy_dispersion.EnergyDispersion2D'>"
    assert str(type(obs.psf)) == "<class 'gammapy.irf.psf_king.PSFKing'>"

    # TODO: Background model loading doesn't work yet
    # ValueError: Expecting X axis in first 2 places, not (DETX_LO, DETX_HI)
    # assert str(type(obs.bkg)) == ""


@requires_data('gammapy-extra')
def test_datastore_load_all(data_store):
    """Test loading data and IRF files via the DataStore"""
    event_lists = data_store.load_all(hdu_class='events')
    assert_allclose(event_lists[0].table['ENERGY'][0], 1.1156039)
    assert_allclose(event_lists[-1].table['ENERGY'][0], 1.0204216)


@requires_data('gammapy-extra')
def test_datastore_obslist(data_store):
    """Test loading data and IRF files via the DataStore"""
    obslist = data_store.obs_list([23523, 23592])
    assert obslist[0].obs_id == 23523

    with pytest.raises(ValueError):
        obslist = data_store.obs_list([11111, 23592])

    obslist = data_store.obs_list([11111, 23523], skip_missing=True)
    assert obslist[0].obs_id == 23523


@requires_data('gammapy-extra')
def test_datastore_subset(tmpdir, data_store):
    """Test creating a datastore as subset of another datastore"""
    selected_obs = data_store.obs_table.select_obs_id([23523, 23592])
    storedir = tmpdir / 'substore'
    data_store.copy_obs(selected_obs, storedir)
    obs_id = [23523, 23592]
    data_store.copy_obs(obs_id, storedir, overwrite=True)

    substore = DataStore.from_dir(storedir)

    assert str(substore.hdu_table.base_dir) == str(storedir)
    assert len(substore.obs_table) == 2

    desired = data_store.obs(23523)
    actual = substore.obs(23523)

    assert str(actual.events.table) == str(desired.events.table)

    # Copy only certain HDU classes
    storedir = tmpdir / 'substore2'
    data_store.copy_obs(obs_id, storedir, hdu_class=['events'])

    substore = DataStore.from_dir(storedir)
    assert len(substore.hdu_table) == 2


@requires_data('gammapy-extra')
def test_data_summary(data_store):
    """Test data summary function"""
    t = data_store.data_summary([23523, 23592])
    assert t[0]['events'] == 620975
    assert t[1]['edisp_2d'] == 28931

    t = data_store.data_summary([23523, 23592], summed=True)
    assert t[0]['psf_3gauss'] == 6042

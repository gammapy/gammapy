# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest, assert_quantity_allclose
from astropy.units import Quantity
from ...data import DataStore, DataManager
from ...utils.testing import data_manager, requires_data, requires_dependency
from ...datasets import gammapy_extra


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_datastore_hd_hap():
    """Test HESS HAP-HD data access."""
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2')

    obs = data_store.obs(obs_id=23523)

    assert str(type((obs.events))) == "<class 'gammapy.data.event_list.EventList'>"
    assert str(type(obs.gti)) == "<class 'gammapy.data.gti.GTI'>"
    assert str(type(obs.aeff)) == "<class 'gammapy.irf.effective_area_table.EffectiveAreaTable2D'>"
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
    assert str(type(obs.aeff)) == "<class 'gammapy.irf.effective_area_table.EffectiveAreaTable2D'>"
    assert str(type(obs.edisp)) == "<class 'gammapy.irf.energy_dispersion.EnergyDispersion2D'>"
    assert str(type(obs.psf)) == "<class 'gammapy.irf.psf_king.PSFKing'>"

    # TODO: Background model loading doesn't work yet
    # ValueError: Expecting X axis in first 2 places, not (DETX_LO, DETX_HI)
    # assert str(type(obs.bkg)) == ""


@requires_data('gammapy-extra')
@requires_dependency('yaml')
def test_datastore_construction():
    """Construct DataStore objects in various ways"""
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
    data_store.info()

    DataManager.DEFAULT_CONFIG_FILE = gammapy_extra.filename('datasets/data-register.yaml')
    data_store = DataStore.from_name('hess-crab4-hd-hap-prod2')
    data_store.info()


@requires_data('gammapy-extra')
@requires_dependency('yaml')
def test_datastore_load_all(data_manager):
    """Test loading data and IRF files via the DataStore"""
    data_store = data_manager['hess-crab4-hd-hap-prod2']
    event_lists = data_store.load_all(hdu_class='events')
    assert_allclose(event_lists[0]['ENERGY'][0], 1.1156039)
    assert_allclose(event_lists[-1]['ENERGY'][0], 1.0204216)

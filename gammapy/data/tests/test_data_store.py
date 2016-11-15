# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.tests.helper import pytest, assert_quantity_allclose
from astropy.coordinates import Angle, SkyCoord
from astropy.units import Quantity
import astropy.units as u
from ...data import DataStore, DataManager, ObservationList
from ...utils.testing import data_manager, requires_data, requires_dependency
from ...utils.energy import Energy
from ...datasets import gammapy_extra
from ...utils.energy import EnergyBounds


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_datastore_hd_hap():
    """Test HESS HAP-HD data access."""
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2')

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


@requires_data('gammapy-extra')
@requires_dependency('yaml')
def test_datastore_subset(tmpdir, data_manager):
    """Test creating a datastore as subset of another datastore"""
    data_store = data_manager['hess-crab4-hd-hap-prod2']
    selected_obs = data_store.obs_table.select_obs_id([23523, 23592])
    storedir = tmpdir / 'substore'
    data_store.copy_obs(selected_obs, storedir)
    obs_id = [23523, 23592]
    data_store.copy_obs(obs_id, storedir, clobber=True)

    substore = DataStore.from_dir(storedir)

    assert str(substore.hdu_table.base_dir) == str(storedir)
    assert len(substore.obs_table) == 2

    desired = data_store.obs(23523)
    actual = substore.obs(23523)

    assert str(actual.events) == str(desired.events)

    # Copy only certain HDU classes
    storedir = tmpdir / 'substore2'
    data_store.copy_obs(obs_id, storedir, hdu_class=['events'])

    substore = DataStore.from_dir(storedir)
    assert len(substore.hdu_table) == 2


@requires_data('gammapy-extra')
@requires_dependency('yaml')
def test_data_summary(data_manager):
    """Test data summary function"""

    data_store = data_manager['hess-crab4-hd-hap-prod2']
    t = data_store.data_summary([23523, 23592])
    assert t[0]['events'] == 620975
    assert t[1]['edisp_2d'] == 28931

    t = data_store.data_summary([23523, 23592], summed=True)
    assert t[0]['psf_3gauss'] == 6042


@requires_dependency('scipy')
@requires_data('gammapy-extra')
@pytest.mark.parametrize("pars,result", [
    (dict(energy=None, theta=None),
     dict(energy_shape=18, theta_shape=300, psf_energy=2.5178505859375 * u.TeV, psf_theta=0.05 * u.deg,
          psf_exposure=Quantity(6878545291473.34, "cm2 s"), psf_value=Quantity(205215.42446175334, "1/sr"))),
    (dict(energy=EnergyBounds.equal_log_spacing(1, 10, 100, "TeV"), theta=None),
     dict(energy_shape=101, theta_shape=300, psf_energy=1.2589254117941673 * u.TeV, psf_theta=0.05 * u.deg,
          psf_exposure=Quantity(4622187644084.735, "cm2 s"), psf_value=Quantity(119662.71915415104, "1/sr"))),
    (dict(energy=None, theta=Angle(np.arange(0, 2, 0.002), 'deg')),
     dict(energy_shape=18, theta_shape=1000, psf_energy=2.5178505859375 * u.TeV, psf_theta=0.02 * u.deg,
          psf_exposure=Quantity(6878545291473.34, "cm2 s"), psf_value=Quantity(23082.369133891403, "1/sr"))),
    (dict(energy=EnergyBounds.equal_log_spacing(1, 10, 100, "TeV"), theta=Angle(np.arange(0, 2, 0.002), 'deg')),
     dict(energy_shape=101, theta_shape=1000, psf_energy=1.2589254117941673 * u.TeV, psf_theta=0.02 * u.deg,
          psf_exposure=Quantity(4622187644084.735, "cm2 s"), psf_value=Quantity(27987.773313506143, "1/sr"))),
])
def test_make_psf(pars, result):
    position = SkyCoord(83.63, 22.01, unit='deg')
    store = gammapy_extra.filename("datasets/hess-crab4-hd-hap-prod2")
    data_store = DataStore.from_dir(store)

    obs1 = data_store.obs(23523)
    psf = obs1.make_psf(position=position, energy=pars["energy"], theta=pars["theta"])

    assert_allclose(psf.offset.shape, result["theta_shape"])
    assert_allclose(psf.energy.shape, result["energy_shape"])
    assert_allclose(psf.exposure.shape, result["energy_shape"])
    assert_allclose(psf.psf_value.shape, (result["energy_shape"], result["theta_shape"]))

    assert_quantity_allclose(psf.offset[10], result["psf_theta"])
    assert_quantity_allclose(psf.energy[10], result["psf_energy"])
    assert_quantity_allclose(psf.exposure[10], result["psf_exposure"])
    assert_quantity_allclose(psf.psf_value[10, 50], result["psf_value"])


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_make_meanedisp(tmpdir):
    position = SkyCoord(83.63, 22.01, unit='deg')
    store = gammapy_extra.filename("datasets/hess-crab4-hd-hap-prod2")
    data_store = DataStore.from_dir(store)

    obs1 = data_store.obs(23523)
    obs2 = data_store.obs(23592)
    obslist = ObservationList([obs1, obs2])

    e_true = EnergyBounds.equal_log_spacing(0.01, 150, 80, "TeV")
    e_reco = EnergyBounds.equal_log_spacing(0.5, 100, 15, "TeV")
    rmf = obslist.make_mean_edisp(position=position, e_true=e_true, e_reco=e_reco)

    assert len(rmf.e_true.nodes) == 80
    assert len(rmf.e_reco.nodes) == 15
    assert_quantity_allclose(rmf.data[53, 8], 0.0559785805550798)

    rmf2 = obslist.make_mean_edisp(position=position, e_true=e_true, e_reco=e_reco,
                                   low_reco_threshold=Energy(1, "TeV"), high_reco_threshold=Energy(60, "TeV"))
    # Test that the energy dispersion is equal to zero for the reco energy bin inferior to low_reco_threshold and
    #  superior to high_reco_threshold
    i2 = np.where(rmf2.data[:, 13] != 0)[0]
    assert len(i2) == 0
    i2 = np.where(rmf2.data[:, 1] != 0)[0]
    assert len(i2) == 0
    # Test that for the reco energy bin inside the thresholds, the edisp defined without threshold in the same than
    # the edisp defined with the low and high reco energy thresholds
    i = np.where(rmf.data[:, 12] != 0)[0]
    i2 = np.where(rmf2.data[:, 12] != 0)[0]
    assert_equal(i, i2)
    i = np.where(rmf.data[:, 2] != 0)[0]
    i2 = np.where(rmf2.data[:, 2] != 0)[0]
    assert_equal(i, i2)

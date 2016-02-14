# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest
from ...data import EventList
from ...data import DataStore, DataManager
from ...utils.testing import requires_data, data_manager
from ...utils.scripts import make_path
from ...datasets import gammapy_extra
from numpy.testing import assert_allclose

@requires_data('gammapy-extra')
def test_DataStore_construction(data_manager):
    """Construct DataStore objects in various ways"""
    data_store = data_manager['hess-crab4-hd-hap-prod2']
    data_store.info()

    DataManager.DEFAULT_CONFIG_FILE = gammapy_extra.filename('datasets/data-register.yaml')
    data_store = DataStore.from_name('hess-crab4-hd-hap-prod2')
    data_store.info()


    base_dir = make_path('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2')
    data_store = DataStore.from_dir(base_dir)


@requires_data('gammapy-extra')
def test_DataStore_filenames(data_manager):
    """Check if filenames are constructed correctly"""
    data_store = data_manager['hess-crab4-hd-hap-prod2']

    filename = data_store.filename(obs_id=23523, filetype='aeff')
    assert filename == str(make_path(
        '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_aeff_2d_023523.fits.gz'))

    filename = data_store.filename(obs_id=23523, filetype='aeff', abspath=False)

    assert filename == 'run023400-023599/run023523/hess_aeff_2d_023523.fits.gz'

    with pytest.raises(IndexError) as exc:
        data_store.filename(obs_id=89565, filetype='aeff')
    msg = 'File not in table: OBS_ID = 89565, TYPE = aeff'
    assert exc.value.args[0] == msg

    with pytest.raises(ValueError):
        data_store.filename(obs_id=89565, filetype='effective area')

@requires_data('gammapy-extra')
def test_DataStore_filenames_pa(data_manager):

    data_store = data_manager['hess-crab4-pa']

    filename = data_store.filename(obs_id=23523, filetype='aeff')

    assert filename == str(make_path(
        '$GAMMAPY_EXTRA/datasets/hess-crab4-pa/run23400-23599/run23523/aeff_2d_23523.fits.gz'))

@requires_data('gammapy-extra')
def test_DataStore_load(data_manager):
    """Test loading data and IRF files via the DataStore"""
    data_store = data_manager['hess-crab4-hd-hap-prod2']

    events = data_store.load(obs_id=23523, filetype='events')
    assert isinstance(events, EventList)


@requires_data('gammapy-extra')
def test_DataStore_load_all(data_manager):
    """Test loading data and IRF files via the DataStore"""
    data_store = data_manager['hess-crab4-hd-hap-prod2']
    event_lists = data_store.load_all(filetype='events')
    assert_allclose(event_lists[0]['ENERGY'][0], 1.1156039)
    assert_allclose(event_lists[-1]['ENERGY'][0], 1.0204216)

@pytest.mark.xfail
@requires_data('gammapy-extra')
def test_DataStore_other(data_manager):
    """Misc tests"""
    data_store = data_manager['hess-hap-hd-prod01-std_zeta_fullEnclosure']

    # run_list_selection = dict(shape='box', frame='galactic',
    #                           lon=(-100, 50), lat=(-5, 5), border=2)
    # run_list = data_store.make_run_list(run_list_selection)
    # print(len(run_list))

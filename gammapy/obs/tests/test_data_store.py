# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest
from ...data import EventList
from ...obs import DataStore
from ...utils.testing import requires_data, data_manager

@pytest.mark.xfail
@requires_data('hess')
def test_DataStore_construction(data_manager):
    """Construct DataStore objects in various ways"""
    data_store = data_manager['hess-hap-hd-prod01-std_zeta_fullEnclosure']
    data_store.info()

    data_store = DataStore.from_name('hess-hap-hd-prod01-std_zeta_fullEnclosure')
    data_store.info()

    base_dir = '/Users/deil/work/_Data/hess/fits/parisanalysis/fits_prod02/pa/Model_Deconvoluted_Prod26/Mpp_Std'
    data_store = DataStore.from_dir(base_dir)

@pytest.mark.xfail
@requires_data('hess')
def test_DataStore_filenames(data_manager):
    """Check if filenames are constructed correctly"""
    data_store = data_manager['hess-hap-hd-prod01-std_zeta_fullEnclosure']

    filename = data_store.filename(obs_id=23037, filetype='aeff')
    assert filename == '/Users/deil/work/_Data/hess/fits/hap-hd/fits_prod01/std_zeta_fullEnclosure/run023000-023199/run023037/hess_aeff_023037.fits.gz'

    filename = data_store.filename(obs_id=23037, filetype='aeff', abspath=False)
    assert filename == 'run023000-023199/run023037/hess_aeff_023037.fits.gz'

    with pytest.raises(IndexError) as exc:
        data_store.filename(obs_id=89565, filetype='aeff')
    msg = 'File not in table: OBS_ID = 89565, TYPE = aeff'
    assert exc.value.args[0] == msg

    with pytest.raises(ValueError):
        data_store.filename(obs_id=89565, filetype='effective area')

    data_store = data_manager['hess-paris-prod02']

    filename = data_store.filename(obs_id=23037, filetype='aeff')
    assert filename == '/Users/deil/work/_Data/hess/fits/parisanalysis/fits_prod02/pa/Model_Deconvoluted_Prod26/Mpp_Std/run023000-023199/run023037/hess_aeff_2d_023037.fits.gz'

    filename = data_store.filename(obs_id=23037, filetype='aeff', abspath=False)
    assert filename == 'run023000-023199/run023037/hess_aeff_2d_023037.fits.gz'

@pytest.mark.xfail
@requires_data('hess')
def test_DataStore_load(data_manager):
    """Test loading data and IRF files via the DataStore"""
    data_store = data_manager['hess-hap-hd-prod01-std_zeta_fullEnclosure']

    events = data_store.load(obs_id=23037, filetype='events')
    assert isinstance(events, EventList)

@pytest.mark.xfail
@requires_data('hess')
def test_DataStore_other(data_manager):
    """Misc tests"""
    data_store = data_manager['hess-hap-hd-prod01-std_zeta_fullEnclosure']

    # run_list_selection = dict(shape='box', frame='galactic',
    #                           lon=(-100, 50), lat=(-5, 5), border=2)
    # run_list = data_store.make_run_list(run_list_selection)
    # print(len(run_list))

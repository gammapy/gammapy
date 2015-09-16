# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest
from ...obs import DataStore, DataStoreIndexTable

HESSFITS_MPP = '/Users/deil/work/_Data/hess/HESSFITS/pa/Model_Deconvoluted_Prod26/Mpp_Std/'


# TODO: fix this test!
@pytest.mark.xfail
def test_DataStoreIndexTable():
    filename = HESSFITS_MPP + '/runinfo.fits'
    table = DataStoreIndexTable.read(filename)
    info_string = table.info()
    print(info_string)
    print(table.radec)


# TODO: fix this test!
@pytest.mark.xfail
def test_DataStore():
    data_store = DataStore(dir=HESSFITS_MPP)
    info_string = data_store.info()
    print(info_string)

    filename = data_store.filename(obs_id=89565, filetype='effective area')
    print(filename)
    assert filename == 'run089400-089599/run089565/hess_aeff_2d_089565.fits.gz'

    run_list_selection = dict(shape='box', frame='galactic',
                              lon=(-100, 50), lat=(-5, 5), border=2)
    run_list = data_store.make_run_list(run_list_selection)
    print(len(run_list))

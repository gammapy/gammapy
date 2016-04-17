"""Some test examples how to access IACT data.
"""
import logging
from gammapy.data import DataManager, DataStore, HDUIndexTable

logging.basicConfig(level=logging.DEBUG)

log = logging.getLogger(__name__)


def test_data_manager():
    filename = '/Users/deil/code/gammapy/gammapy/data/tests/data/data-register.yaml'
    data_manager = DataManager.from_yaml(filename)
    data_manager.info()

    # ds = dm['hess-hap-prod01']
    # ds = dm['hess-hap-hd-prod01-std_zeta_fullEnclosure']

    for data_store in data_manager.stores:
        data_store.info()


def test_obs():
    # data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2')
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-pa')
    data_store.info()
    obs = data_store.obs(obs_id=23523)
    # import IPython; IPython.embed()

    # obs.info()
    # print(type(obs.events))
    # print(type(obs.gti))
    print(type(obs.aeff))
    print(type(obs.edisp))
    print(type(obs.psf))
    # print(type(obs.bkg))


def test_hdu_index():
    hdu_index = HDUIndexTable.read('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/hdu-index.fits.gz')
    hdu_index.info()
    location = hdu_index.hdu_location(obs_id=23523, hdu_type='events')
    location.info()


    # hdu_index = HDUIndexTable.read('$GAMMAPY_EXTRA/datasets/hess-crab4-pa/hdu-index.fits.gz')
    # hdu_index.info()
    # location = hdu_index.hdu_location(obs_id=23523, hdu_type='psf')
    # location.info()


# def test_high_level():
#     data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2')
#     obs_ids = data_store.obs_table['OBS_ID'][:3]
#
#     counts_cube = SkyCube('binning specification')
#
#     for obs_id in obs_ids:
#         obs =
#         counts_cube.fill(obs.counts)
#
#
#     psfs = [data_store.obs(obs_id).load(hdu_class='psf_king') for obs_id in obs_ids]
#     print psfs

if __name__ == '__main__':
    # test_data_manager()
    test_obs()
    # test_hdu_index()
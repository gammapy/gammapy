"""Some test examples how to access IACT data.
"""
import logging
from gammapy.data import DataManager, DataStore
from gammapy.datasets import gammapy_extra

logging.basicConfig(level=logging.DEBUG)

log = logging.getLogger(__name__)


def test_data_manager():
    filename = '/Users/deil/code/gammapy/gammapy/obs/tests/data/data-register.yaml'
    data_manager = DataManager.from_yaml(filename)
    data_manager.info()

    # ds = dm['hess-hap-prod01']
    # ds = dm['hess-hap-hd-prod01-std_zeta_fullEnclosure']

    for data_store in data_manager.stores:
        data_store.info()


def test_data_store():
    data_manager = DataManager()
    data_store = data_manager['hess-hap-hd-prod01-std_zeta_fullEnclosure']
    print(data_store.filename(obs_id=23037, filetype='events'))
    events = data_store.load(obs_id=23037, filetype='events')
    print(events.__class__)


def test_data_store2():
    dir = gammapy_extra.dir / 'datasets/hess-crab4-hd-hap-prod2'
    ds = DataStore.from_dir(dir)
    ds.info()


if __name__ == '__main__':
    # test_data_manager()
    # test_data_store()
    test_data_store2()

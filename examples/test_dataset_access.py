"""Examples of downloading / accessing datasets in Gammapy.

TODO: move this to `gammapy/datasets/tests/test_core.py`
"""
import logging
from gammapy.datasets import Datasets

logging.basicConfig(level=logging.DEBUG)

log = logging.getLogger(__name__)


def test_info():
    Datasets().info()


def test_fetch():
    Datasets().fetch_one(name='catalog-atnf')
    Datasets().fetch_all(tags='catalog')


def test_access():
    print(Datasets()['catalog-atnf'].filename)


if __name__ == '__main__':
    test_info()
    test_fetch()
    test_access()

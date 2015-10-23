"""Examples of downloading / accessing datasets in Gammapy.
"""
from pprint import pprint
import logging
from gammapy.datasets import Datasets

logging.basicConfig(level=logging.DEBUG)

log = logging.getLogger(__name__)


def test_info():
    Datasets().info()
    # import IPython; IPython.embed()


def test_fetch():
    # import IPython; IPython.embed()
    # Datasets().fetch_one(name='catalog-atnf')
    Datasets().fetch_all(tags='catalog')


def test_access():
    print(Datasets()['catalog-atnf'].filename)

    # Datasets().load('catalog-atnf')



if __name__ == '__main__':
    # test_info()
    # test_fetch()
    test_access()


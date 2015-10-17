import os
from astropy.utils.data import get_pkg_data_filename
from astropy.tests.helper import pytest
from ..data_manager import DataManager

# For now the tests use hardcoded paths to data locations on on my computer.
# TODO: set up some public test data (or simulate some) for testing
if os.environ['USER'] == 'deil':
    _HAS_TEST_DATA = True
else:
    _HAS_TEST_DATA = False


@pytest.fixture
def data_manager():
    filename = get_pkg_data_filename('data/data-register.yaml')
    return DataManager.from_yaml(filename)


# TODO: create one or two datastores with simulated data and use that
# for many Gammapy tests
# https://pytest.org/latest/tmpdir.html#the-tmpdir-factory-fixture

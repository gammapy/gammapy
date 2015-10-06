# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_equal, assert_almost_equal
from astropy.io import fits
from astropy.tests.helper import remote_data
from astropy.tests.helper import pytest
from ...scripts import GammapySpectrumAnalysis
from ...datasets import get_path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

@remote_data
@pytest.mark.skipif('not HAS_YAML')
def test_spectrum_pipe(tmpdir):

    configfile = get_path('../test_datasets/scripts/spectrum_pipe_example.yaml',
                          location='remote')
    analysis = GammapySpectrumAnalysis.from_yaml(configfile)

    # TODO: test more stuff once the DataStore class can be accessed remotely

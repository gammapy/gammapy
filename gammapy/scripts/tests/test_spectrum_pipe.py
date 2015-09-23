# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_equal, assert_almost_equal
from astropy.io import fits
from astropy.tests.helper import remote_data
from gammapy.scripts import GammapySpectrumAnalysis
from ...datasets import get_path
import yaml

def test_spectrum_pipe(tmpdir):

    #Change to remote file in the end
    configfile = "~/Software/gammapy/gammapy/scripts/spectrum_pipe_example.yaml"
    analysis = GammapySpectrumAnalysis(configfile)

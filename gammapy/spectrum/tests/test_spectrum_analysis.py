# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.tests.helper import pytest
from ...utils.testing import requires_dependency, requires_data
from ...datasets import gammapy_extra
from ...spectrum.spectrum_analysis import SpectrumAnalysis


# TODO: this test is currently broken ... fix it!
@pytest.mark.xfail
@requires_dependency('yaml')
@requires_data('gammapy-extra')
def test_spectrum_analysis(tmpdir):

    configfile = gammapy_extra.filename('test_datasets/spectrum/spectrum_analysis_example.yaml')
    analysis = SpectrumAnalysis.from_yaml(configfile)

    # TODO: test more stuff once the DataStore class can be accessed remotely

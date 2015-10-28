# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.tests.helper import pytest
from ...utils.testing import requires_dependency, requires_data, data_manager
from ...datasets import gammapy_extra
from ...spectrum.spectrum_analysis import SpectrumAnalysis


@requires_dependency('yaml')
@requires_data('gammapy-extra')
@requires_data('hess')
def test_spectrum_analysis(data_manager):

    configfile = gammapy_extra.filename('test_datasets/spectrum/spectrum_analysis_example.yaml')

    analysis = SpectrumAnalysis.from_yaml(configfile)
    analysis.run()
    

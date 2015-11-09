# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.tests.helper import pytest
from ...utils.testing import requires_dependency, requires_data
from numpy.testing import assert_allclose
from ...datasets import gammapy_extra
from ...spectrum.spectrum_analysis import SpectrumAnalysis


@requires_dependency('yaml')
@requires_data('gammapy-extra')
@requires_data('hess')
def test_spectrum_analysis(tmpdir):

    configfile = gammapy_extra.filename('test_datasets/spectrum/spectrum_analysis_example_ring.yaml')
    analysis = SpectrumAnalysis.from_yaml(configfile)
    fit = analysis.run()
    assert_allclose(fit['parvals'][0], 2.44, rtol=1e-2)

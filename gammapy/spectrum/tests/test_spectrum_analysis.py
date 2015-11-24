# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.tests.helper import pytest
from numpy.testing import assert_allclose
from ...utils.testing import requires_dependency, requires_data
from ...datasets import gammapy_extra
from ...utils.scripts import read_yaml
from ...spectrum.spectrum_analysis import (
    SpectrumAnalysis,
    run_spectrum_analysis_using_config,
)

@requires_dependency('yaml')
@requires_data('gammapy-extra')
def test_spectrum_analysis_from_configfile(tmpdir):

    configfile = gammapy_extra.filename('test_datasets/spectrum/spectrum_analysis_example.yaml')

    import yaml
    config = read_yaml(configfile)
    config['general']['outdir'] = str(tmpdir)
    
    ana = run_spectrum_analysis_using_config(config)
    assert_allclose(ana.fit['parvals'][0], 2.24, rtol = 1e-2)

    config['off_region']['type'] = 'reflected'
    ana = run_spectrum_analysis_using_config(config)
    assert_allclose(ana.fit['parvals'][0], 2.24, rtol = 1e-2)


# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.tests.helper import pytest
from ...utils.testing import requires_dependency, requires_data
from numpy.testing import assert_allclose
from ...datasets import gammapy_extra
from ...spectrum.spectrum_analysis import (
    SpectrumAnalysis,
    run_spectrum_analysis_using_configfile,
    run_spectrum_analysis_using_config)

@requires_dependency('yaml')
@requires_data('gammapy-extra')
@requires_data('hess')
def test_spectrum_analysis_from_configfile(tmpdir):
    import yaml

    configfile = gammapy_extra.filename('test_datasets/spectrum/spectrum_analysis_example_ring.yaml')

    import yaml
    with open(configfile) as fh:
        config = yaml.safe_load(fh)

    config['general']['outdir']=str(tmpdir)

    fit = run_spectrum_analysis_using_config(config)

    assert_allclose(fit['parvals'][0], 2.44, rtol = 1e-2)

    config['off_region']['type'] = 'reflected'

    fit = run_spectrum_analysis_using_config(config)
    assert_allclose(fit['parvals'][0], 2.2, rtol = 1e-2)


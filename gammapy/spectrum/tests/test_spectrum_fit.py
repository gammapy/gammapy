# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.tests.helper import pytest
from numpy.testing import assert_allclose

from gammapy.spectrum import run_spectrum_extraction_using_config
from ...datasets import gammapy_extra
from ...spectrum.spectrum_fit import (
    SpectrumFit,
    run_spectrum_fit_using_config,
)
from ...spectrum.results import SpectrumFitResult
from ...utils.scripts import read_yaml
from ...utils.testing import requires_dependency, requires_data, SHERPA_LT_4_8
from astropy.utils.compat import NUMPY_LT_1_9

@pytest.mark.skipif('NUMPY_LT_1_9')
@pytest.mark.skipif('SHERPA_LT_4_8')
@requires_dependency('sherpa')
@requires_data('gammapy-extra')
def test_spectral_fit():
    pha1 = gammapy_extra.filename("datasets/hess-crab4_pha/pha_run23592.pha")
    pha2 = gammapy_extra.filename("datasets/hess-crab4_pha/pha_run23526.pha")
    pha_list = [pha1, pha2]
    fit = SpectrumFit(pha_list)
    fit.model = 'PL'
    fit.energy_threshold_low = '100 GeV'
    fit.energy_threshold_high = '10 TeV'
    fit.run(method='sherpa')
    assert fit.result.spectral_model == 'PowerLaw'


def test_spectral_fit_using_config(tmpdir):

    configfile = gammapy_extra.filename(
        'test_datasets/spectrum/spectrum_analysis_example.yaml')
    config = read_yaml(configfile)

    config['extraction']['results']['outdir'] = str(tmpdir)

    run_spectrum_extraction_using_config(config)

    config['fit']['observation_table'] = str(tmpdir / 'observations.fits')
    config['fit']['outdir'] = str(tmpdir)
    tmpfile = tmpdir / 'fit.yaml'
    config['fit']['result_file'] = str(tmpfile)

    run_spectrum_fit_using_config(config)

    actual = SpectrumFitResult.from_yaml(str(tmpfile))
    desired = SpectrumFitResult.from_yaml(
        gammapy_extra.filename('test_datasets/spectrum/fitfunction.yaml'))

    # Todo Actually compare the two files. Not possible now due to float issues
    assert_allclose(actual.parameters.index.value, desired.parameters.index.value)

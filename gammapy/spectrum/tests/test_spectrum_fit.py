# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.tests.helper import pytest

from ...datasets import gammapy_extra
from ...spectrum.spectrum_extraction import SpectrumObservationList, SpectrumObservation
from ...spectrum.spectrum_fit import SpectrumFit
from ...utils.testing import requires_dependency, requires_data, SHERPA_LT_4_8

@pytest.mark.skipif('NUMPY_LT_1_9')
@pytest.mark.skipif('SHERPA_LT_4_8')
@requires_dependency('sherpa')
@requires_data('gammapy-extra')
def test_spectral_fit():
    pha1 = gammapy_extra.filename("datasets/hess-crab4_pha/pha_run23592.pha")
    pha2 = gammapy_extra.filename("datasets/hess-crab4_pha/pha_run23526.pha")

    obs1 = SpectrumObservation.read_ogip(pha1)
    obs2 = SpectrumObservation.read_ogip(pha2)
    obs_list = SpectrumObservationList([obs1, obs2])
    fit = SpectrumFit(obs_list)
    fit.model = 'PL'
    fit.energy_threshold_low = '100 GeV'
    fit.energy_threshold_high = '10 TeV'
    fit.run(method='sherpa')
    assert fit.result.spectral_model == 'PowerLaw'



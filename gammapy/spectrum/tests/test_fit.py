# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest, assert_quantity_allclose
import astropy.units as u
from astropy.utils.compat import NUMPY_LT_1_9
from numpy.testing import assert_allclose
from ...datasets import gammapy_extra
from ...spectrum import (
    SpectrumObservationList,
    SpectrumObservation,
    SpectrumFit,
    SpectrumFitResult,
    models,
)
from ...utils.testing import requires_dependency, requires_data, SHERPA_LT_4_8


@pytest.mark.skipif('NUMPY_LT_1_9')
@pytest.mark.skipif('SHERPA_LT_4_8')
@requires_dependency('sherpa')
@requires_data('gammapy-extra')
def test_spectral_fit(tmpdir):
    pha1 = gammapy_extra.filename("datasets/hess-crab4_pha/pha_obs23592.fits")
    pha2 = gammapy_extra.filename("datasets/hess-crab4_pha/pha_obs23523.fits")
    obs1 = SpectrumObservation.read(pha1)
    obs2 = SpectrumObservation.read(pha2)
    obs_list = SpectrumObservationList([obs1, obs2])

    model = models.PowerLaw(index = 2 * u.Unit(''),
                            amplitude = 10 ** -12 * u.Unit('cm-2 s-1 TeV-1'),
                            reference = 1 * u.TeV)

    # Test obs list and assert on results
    fit = SpectrumFit(obs_list, model)
    fit.run(outdir=tmpdir)

    # Make sure FitResult is correctly readable
    read_result = SpectrumFitResult.from_yaml(tmpdir / 'fit_result_PowerLaw.yaml')
    test_e = 12.5 * u.TeV
    assert_quantity_allclose(fit.result[0].fit.model(test_e),
                             read_result.model(test_e))

    result = fit.result[0]
    assert 'PowerLaw' in str(result.fit)
    assert_allclose(result.fit.statval, 103.595, rtol=1e-3)
    assert_quantity_allclose(result.fit.model.parameters.index,
                             2.116 * u.Unit(''), rtol=1e-3)
    model_with_errors = result.fit.model_with_uncertainties
    assert_allclose(model_with_errors.parameters.index.s,
                    0.0543, rtol=1e-3)

    # Actual fit range can differ from threshold due to binning effects
    # We take the lowest bin that is completely within threshold
    # Sherpa quotes the lincenter of that bin as fitrange
    thres_bin = obs1.on_vector.energy.find_node(obs1.lo_threshold)
    desired = obs1.on_vector.energy.lin_center()[thres_bin + 1]
    actual = result.fit.fit_range[0]
    assert_quantity_allclose(actual, desired)

    # Test npred
    npred = obs1.predicted_counts(result.fit.model)
    assert_allclose(result.fit.npred, npred.data, rtol=1e-3)

    # Restrict fit range
    fit_range = [4, 20] * u.TeV
    fit.fit_range = fit_range
    fit.run()

    range_bin = obs1.on_vector.energy.find_node(fit_range[1])
    desired = obs1.on_vector.energy.lin_center()[range_bin]
    actual = fit.result[0].fit.fit_range[1]
    assert_quantity_allclose(actual, desired)

    # Make sure fit range is not extended below threshold
    fit_range = [0.001, 10] * u.TeV
    fit.fit_range = fit_range
    fit.run()
    desired = obs1.on_vector.energy.lin_center()[thres_bin + 1]
    actual = fit.result[0].fit.fit_range[0]

    assert_quantity_allclose(actual, desired)

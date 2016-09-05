# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest, assert_quantity_allclose
import astropy.units as u
import numpy as np
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
from ...utils.testing import (
    requires_dependency,
    requires_data,
    SHERPA_LT_4_8,
)


@pytest.mark.skipif('NUMPY_LT_1_9')
@pytest.mark.xfail(reason='wait for https://github.com/sherpa/sherpa/pull/249')
@requires_dependency('sherpa')
@requires_dependency('matplotlib')
@requires_data('gammapy-extra')
def test_spectral_fit(tmpdir):
    pha1 = gammapy_extra.filename("datasets/hess-crab4_pha/pha_obs23592.fits")
    pha2 = gammapy_extra.filename("datasets/hess-crab4_pha/pha_obs23523.fits")
    obs1 = SpectrumObservation.read(pha1)
    obs2 = SpectrumObservation.read(pha2)
    obs_list = SpectrumObservationList([obs1, obs2])

    model = models.PowerLaw(index=2 * u.Unit(''),
                            amplitude=10 ** -12 * u.Unit('cm-2 s-1 TeV-1'),
                            reference=1 * u.TeV)

    fit = SpectrumFit(obs_list, model)
    fit.run(outdir=tmpdir)

    # Make sure FitResult is correctly readable
    read_result = SpectrumFitResult.from_yaml(tmpdir / 'fit_result_PowerLaw.yaml')
    test_e = 12.5 * u.TeV
    assert_quantity_allclose(fit.result[0].model(test_e),
                             read_result.model(test_e))

    result = fit.result[0]
    result.plot()

    # Test various methods
    assert 'PowerLaw' in str(result)
    assert 'index' in result.to_table().colnames

    # Test values
    assert_allclose(result.statval, 103.595, rtol=1e-3)
    assert_quantity_allclose(result.model.parameters.index,
                             2.116 * u.Unit(''), rtol=1e-3)
    model_with_errors = result.model_with_uncertainties
    assert_allclose(model_with_errors.parameters.index.s,
                    0.0542, rtol=1e-3)

    # Actual fit range can differ from threshold due to binning effects
    # We take the lowest bin that is completely within threshold
    # Sherpa quotes the lincenter of that bin as fitrange
    thres_bin = obs1.on_vector.energy.find_node(obs1.lo_threshold)
    desired = obs1.on_vector.energy.lin_center()[thres_bin + 1]
    actual = result.fit_range[0]
    assert_quantity_allclose(actual, desired)

    # Test npred
    npred = obs1.predicted_counts(result.model)
    assert_allclose(result.npred, npred.data, rtol=1e-3)

    # Restrict fit range
    fit_range = [4, 20] * u.TeV
    fit.fit_range = fit_range
    fit.run()

    range_bin = obs1.on_vector.energy.find_node(fit_range[1])
    desired = obs1.on_vector.energy.lin_center()[range_bin]
    actual = fit.result[0].fit_range[1]
    assert_quantity_allclose(actual, desired)

    # Make sure fit range is not extended below threshold

    fit_range = [0.001, 10] * u.TeV
    fit.fit_range = fit_range
    fit.run()
    desired = obs1.on_vector.energy.lin_center()[thres_bin + 1]
    actual = fit.result[0].fit_range[0]

    assert_quantity_allclose(actual, desired)

    # Test fluxpoints computation
    binning = np.logspace(0, 1, 5) * u.TeV
    result = fit.compute_fluxpoints(binning=binning)
    result.plot(energy_range=binning[[0, -1]])
    actual = result.points['DIFF_FLUX'].quantity[2]
    desired = 1.118e-12 * u.Unit('cm-2 s-1 TeV-1')
    assert_quantity_allclose(actual, desired, rtol=1e-3)

    actual = result.points['DIFF_FLUX_ERR_HI'].quantity[2]
    desired = 1.842e-13 * u.Unit('cm-2 s-1 TeV-1')
    assert_quantity_allclose(actual, desired, rtol=1e-3) 

    residuals = result.flux_point_residuals
    assert_allclose(residuals[2].s, 0.1905, rtol=1e-3) 
    assert_allclose(residuals[2].s, 0.0938, rtol=1e-3)

    # Test ECPL
    ecpl = models.ExponentialCutoffPowerLaw(
        index=2 * u.Unit(''),
        amplitude=10 ** -12 * u.Unit('cm-2 s-1 TeV-1'),
        reference=1 * u.TeV,
        lambda_=0.1 / u.TeV
    )

    fit = SpectrumFit(obs_list, ecpl)
    fit.fit()
    assert_quantity_allclose(fit.result[0].model.parameters.lambda_,
                             0.06321 / u.TeV, rtol=1e-3)


@requires_dependency('sherpa')
@pytest.mark.skipif('NUMPY_LT_1_9')
@pytest.mark.xfail(reason = 'wait for https://github.com/sherpa/sherpa/pull/249')
@requires_data('gammapy-extra')
def test_stacked_fit():
    pha1 = gammapy_extra.filename("datasets/hess-crab4_pha/pha_obs23592.fits")
    pha2 = gammapy_extra.filename("datasets/hess-crab4_pha/pha_obs23523.fits")
    obs1 = SpectrumObservation.read(pha1)
    obs2 = SpectrumObservation.read(pha2)

    stacked_obs = SpectrumObservation.stack([obs1, obs2])

    model = models.PowerLaw(index=2 * u.Unit(''),
                            amplitude=10 ** -12 * u.Unit('cm-2 s-1 TeV-1'),
                            reference=1 * u.TeV)

    fit = SpectrumFit(stacked_obs, model)
    fit.fit()
    pars = fit.global_result.model.parameters
    assert_quantity_allclose(pars.index, 2.12, rtol=1e-2)
    assert_quantity_allclose(pars.amplitude,
                             2.11e-11 * u.Unit('cm-2 s-1 TeV-1'),
                             rtol=1e-2)


@requires_dependency('sherpa')
@requires_data('gammapy-extra')
def test_sherpa_fit(tmpdir):
    # this is to make sure that the written PHA files work with sherpa
    pha1 = gammapy_extra.filename("datasets/hess-crab4_pha/pha_obs23592.fits")

    import sherpa.astro.ui as sau
    from sherpa.models import PowLaw1D
    sau.load_pha(pha1)
    sau.set_stat('wstat')
    model = PowLaw1D('powlaw1d.default')
    model.ref = 1e9
    model.ampl = 1
    model.gamma = 2
    sau.set_model(model * 1e-20)
    sau.fit()
    assert_allclose(model.pars[0].val, 2.0198, atol=1e-4)
    assert_allclose(model.pars[2].val, 2.3564, atol=1e-4)

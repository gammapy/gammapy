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
@requires_dependency('sherpa')
@requires_data('gammapy-extra')
class TestSpectralFit:

    def setup(self):
        self.obs_list = SpectrumObservationList.read(
            '$GAMMAPY_EXTRA/datasets/hess-crab4_pha')

        self.pwl = models.PowerLaw(index=2 * u.Unit(''),
                                   amplitude=10 ** -12 * u.Unit('cm-2 s-1 TeV-1'),
                                   reference=1 * u.TeV)

        self.ecpl = models.ExponentialCutoffPowerLaw(
            index=2 * u.Unit(''),
            amplitude=10 ** -12 * u.Unit('cm-2 s-1 TeV-1'),
            reference=1 * u.TeV,
            lambda_=0.1 / u.TeV
        )

        # Example fit for one observation
        self.fit = SpectrumFit(self.obs_list[0], self.pwl)
        self.fit.fit()
        self.result = self.fit.result[0]

    def test_basic_results(self):
        assert self.fit.method_fit.name == 'simplex'
        assert_allclose(self.result.statval, 34.19706702533566)
        pars = self.result.model.parameters
        assert_quantity_allclose(pars.index,
                                 2.23957544167327)
        assert_quantity_allclose(pars.amplitude,
                                 2.018513315748709e-07 * u.Unit('m-2 s-1 TeV-1'))
        par_errors = self.result.model_with_uncertainties.parameters
        assert_allclose(par_errors.index.s, 0.09558428890966723)
        assert_allclose(par_errors.amplitude.s, 2.2154024177186417e-08)

    def test_npred(self):
        actual = self.result.obs.predicted_counts(self.result.model).data.value
        desired = self.result.npred
        assert_allclose(actual, desired)

    def test_fit_range(self):
        # Actual fit range can differ from threshold due to binning effects
        # We take the lowest bin that is completely within threshold
        # Sherpa quotes the lincenter of that bin as fitrange
        obs = self.result.obs
        thres_bin = obs.on_vector.energy.find_node(obs.lo_threshold)
        desired = obs.on_vector.energy.lin_center()[thres_bin + 1]
        actual = self.result.fit_range[0]
        assert_quantity_allclose(actual, desired)

        # Restrict fit range
        fit_range = [4, 20] * u.TeV
        self.fit.fit_range = fit_range
        self.fit.fit()

        range_bin = obs.on_vector.energy.find_node(fit_range[1])
        desired = obs.on_vector.energy.lin_center()[range_bin]
        actual = self.fit.result[0].fit_range[1]
        assert_quantity_allclose(actual, desired)

        # Make sure fit range is not extended below threshold
        fit_range = [0.001, 10] * u.TeV
        self.fit.fit_range = fit_range
        self.fit.fit()
        desired = obs.on_vector.energy.lin_center()[thres_bin + 1]
        actual = self.fit.result[0].fit_range[0]

        assert_quantity_allclose(actual, desired)

    def test_fit_method(self):
        self.fit.method_fit = "levmar"
        assert self.fit.method_fit.name == "levmar"
        self.fit.fit()
        result = self.fit.result[0]
        assert_quantity_allclose(result.model.parameters.index,
                                 2.2395184727047788)

    def test_ecpl_fit(self):
        fit = SpectrumFit(self.obs_list[0], self.ecpl)
        fit.fit()
        assert_quantity_allclose(fit.result[0].model.parameters.lambda_,
                                 0.028606845248390498 / u.TeV)

    def test_joint_fit(self):
        fit = SpectrumFit(self.obs_list, self.pwl)
        fit.fit()
        assert_quantity_allclose(fit.global_result.model.parameters.index,
                                 2.207512847977245)
        assert_quantity_allclose(fit.global_result.model.parameters.amplitude,
                                 2.3755942722352085e-07 * u.Unit('m-2 s-1 TeV-1'))

    def test_stacked_fit(self):
        stacked_obs = self.obs_list.stack()
        fit = SpectrumFit(stacked_obs, self.pwl)
        fit.fit()
        pars = fit.global_result.model.parameters
        assert_quantity_allclose(pars.index, 2.2462501437579476)
        assert_quantity_allclose(pars.amplitude,
                                 2.5160334568171844e-11 * u.Unit('cm-2 s-1 TeV-1'))


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

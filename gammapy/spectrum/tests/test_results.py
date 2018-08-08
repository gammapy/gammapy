# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from ...utils.testing import assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data
from ..models import PowerLaw
from .. import SpectrumObservation, SpectrumFitResult


@requires_dependency('scipy')
@requires_data('gammapy-extra')
class TestSpectrumFitResult:
    def setup(self):
        filename = "$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23592.fits"
        self.obs = SpectrumObservation.read(filename)
        self.best_fit_model = PowerLaw(index=2 * u.Unit(''),
                                       amplitude=1e-11 * u.Unit('cm-2 s-1 TeV-1'),
                                       reference=1 * u.TeV)
        self.npred = self.obs.predicted_counts(self.best_fit_model).data.data.value
        covar = np.diag([0.1 ** 2, 1e-12 ** 2, 0])
        self.best_fit_model.parameters.covariance = covar
        self.fit_range = [0.1, 50] * u.TeV
        self.fit_result = SpectrumFitResult(
            model=self.best_fit_model,
            fit_range=self.fit_range,
            statname='wstat',
            statval=42,
            npred_src=self.npred,
            npred_bkg=self.npred * 0.5,
            obs=self.obs,
        )

    @requires_dependency('uncertainties')
    def test_basic(self):
        assert 'PowerLaw' in str(self.fit_result)
        assert 'index' in self.fit_result.to_table().colnames

    @requires_dependency('yaml')
    def test_io(self, tmpdir):
        filename = tmpdir / 'test.yaml'
        self.fit_result.to_yaml(filename)
        read_result = SpectrumFitResult.from_yaml(filename)
        test_e = 12.5 * u.TeV
        assert_quantity_allclose(self.fit_result.model(test_e),
                                 read_result.model(test_e))

    @requires_dependency('matplotlib')
    def test_plot(self):
        self.fit_result.plot()

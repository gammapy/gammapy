# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest, assert_quantity_allclose
import astropy.units as u
import numpy as np
from ...datasets import gammapy_extra
from gammapy.spectrum import (
    SpectrumObservation,
    SpectrumFitResult,
    models,
)

from ...utils.testing import (
    requires_dependency,
    requires_data,
)


@requires_dependency('scipy')
@requires_data('gammapy-extra')
class TestSpectrumFitResult:

    def setup(self):
        pha = gammapy_extra.filename("datasets/hess-crab4_pha/pha_obs23592.fits")
        self.obs = SpectrumObservation.read(pha)
        self.best_fit_model = models.PowerLaw(index=2 * u.Unit(''),
                                              amplitude=1e-11 * u.Unit('cm-2 s-1 TeV-1'),
                                              reference=1 * u.TeV)
        self.npred = self.obs.predicted_counts(self.best_fit_model).data.value
        self.covar_axis = ['index', 'amplitude']
        self.covar = np.diag([0.1 ** 2, 1e-12 ** 2])
        self.fit_range = [0.1, 50] * u.TeV
        self.fit_result = SpectrumFitResult(
            model=self.best_fit_model,
            covariance=self.covar,
            covar_axis=self.covar_axis,
            fit_range=self.fit_range,
            statname='wstat',
            statval=42,
            npred=self.npred,
            obs=self.obs,
        )

    def test_basic(self):
        assert 'PowerLaw' in str(self.fit_result)
        assert 'index' in self.fit_result.to_table().colnames

    @requires_dependency('matplotlib')
    def test_plot(self):
        self.fit_result.plot()

    def test_io(self, tmpdir):
        self.fit_result.to_yaml(tmpdir / 'test.yaml')
        read_result = SpectrumFitResult.from_yaml(tmpdir / 'test.yaml')
        test_e = 12.5 * u.TeV
        assert_quantity_allclose(self.fit_result.model(test_e),
                                 read_result.model(test_e))

    def test_model_with_uncertainties(self):
        actual = self.fit_result.model_with_uncertainties.parameters.index.s
        desired = np.sqrt(self.covar[0][0])
        assert actual == desired

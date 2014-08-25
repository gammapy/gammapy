# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.table import Table
from astropy.modeling.models import Gaussian1D
from ...background.models import GaussianBand2D

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
class TestGaussianBand2D():

    def setup(self):
        table = Table()
        table['x'] = [-30, -10, 10, 20]
        table['amplitude'] = [0, 1, 10, 0]
        table['mean'] = [-1, 0, 1, 0]
        table['stddev'] = [0.4, 0.5, 0.3, 1.0]
        self.table = table
        self.model = GaussianBand2D(table)

    def test_eval(self):
        x = np.linspace(-100, 20, 5)
        y = np.linspace(-2, 2, 7)
        x, y = np.meshgrid(x, y)
        image = self.model.eval(x, y)
        assert_allclose(image.sum(), 1.223962643740966)

    def test_parvals(self):
        par = self.model.parvals(-30)
        assert_allclose(par['amplitude'], 0)
        assert_allclose(par['mean'], -1)
        assert_allclose(par['stddev'], 0.4)

    def test_y_model(self):
        model = self.model.y_model(-30)
        assert isinstance(model, Gaussian1D)
        assert_allclose(model.parameters, [0, -1, 0.4])

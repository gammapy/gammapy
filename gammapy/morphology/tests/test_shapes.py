# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import itertools

import numpy as np
from numpy.testing import utils


from astropy.modeling import fitting
from astropy.tests.helper import pytest
from astropy.convolution.utils import discretize_model

from ..shapes import Gaussian2D, Sphere2D, Shell2D, Delta2D

try:
    from scipy import optimize  # pylint: disable=W0611
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

modes = ['center', 'linear_interp', 'oversample']

models_2D = {

    Sphere2D: {
        'parameters': [1, 0, 0, 10],
        'x_values': [0, 10, 5],
        'y_values': [0, 10, 0],
        'z_values': [20, 0, 20 * np.sqrt(0.75)],
        'x_lim': [-11, 11],
        'y_lim': [-11, 11],
        'integral': 4. / 3 * np.pi * 10 ** 3,
    },

    Delta2D: {
        'parameters': [1, 0, 0],
        'x_values': [0, 0.5, -0.5, 0.25, -0.25],
        'y_values': [0, 0.5, -0.5, 0.25, -0.25],
        'z_values': [1, 1, 0, 1, 1],
        'x_lim': [-10, 10],
        'y_lim': [-10, 10],
        'integral': 1,
    },

    Shell2D: {
        'parameters': [1, 0, 0, 9, 10],
        'x_values': [0],
        'y_values': [0],
        'z_values': [1],
        'x_lim': [-11, 11],
        'y_lim': [-11, 11],
        'integral': 2 * np.pi / 3 * (10 ** 3 - 9 ** 3),
    }
}


class TestMorphologyModels(object):
    """
    Test class for all parametric models.

    Test values have to be defined in example_models.py. It currently test the model
    with different input types, evaluates the model at different positions and
    assures that it gives the correct values. And tests if the  model works with
    the NonLinearFitter.
    """

    def setup_class(self):
        self.N = 100
        self.M = 100
        self.eval_error = 0.0001
        self.fit_error = 0.01
        self.x = 5.3
        self.y = 6.7
        self.x1 = np.arange(1, 10, .1)
        self.y1 = np.arange(1, 10, .1)
        self.y2, self.x2 = np.mgrid[:10, :8]

    @pytest.mark.parametrize(('model_class'), models_2D.keys())
    def test_input2D(self, model_class):
        """
        Test model with different input types.
        """
        parameters = models_2D[model_class]['parameters']
        model = create_model(model_class, parameters)
        model(self.x, self.y)
        model(self.x1, self.y1)
        model(self.x2, self.y2)

    @pytest.mark.parametrize(('model_class'), models_2D.keys())
    def test_eval2D(self, model_class):
        """
        Test model values add certain given points
        """
        parameters = models_2D[model_class]['parameters']
        model = create_model(model_class, parameters)
        x = models_2D[model_class]['x_values']
        y = models_2D[model_class]['y_values']
        z = models_2D[model_class]['z_values']
        utils.assert_allclose(model(x, y), z, self.eval_error)

    @pytest.mark.skipif('not HAS_SCIPY')
    @pytest.mark.parametrize(('model_class'), models_2D.keys())
    def test_fitter2D(self, model_class):
        """
        Test if the parametric model works with the fitter.
        """
        x_lim = models_2D[model_class]['x_lim']
        y_lim = models_2D[model_class]['y_lim']

        parameters = models_2D[model_class]['parameters']
        model = create_model(model_class, parameters)

        if isinstance(parameters, dict):
            parameters = [parameters[name] for name in model.param_names]

        if "log_fit" in models_2D[model_class]:
            if models_2D[model_class]['log_fit']:
                x = np.logspace(x_lim[0], x_lim[1], self.N)
                y = np.logspace(y_lim[0], y_lim[1], self.N)
        else:
            x = np.linspace(x_lim[0], x_lim[1], self.N)
            y = np.linspace(y_lim[0], y_lim[1], self.N)
        xv, yv = np.meshgrid(x, y)

        np.random.seed(0)
        # add 10% noise to the amplitude
        data = model(xv, yv) + 0.1 * parameters[0] * (np.random.rand(self.N, self.N) - 0.5)
        fitter = fitting.NonLinearLSQFitter()
        new_model = fitter(model, xv, yv, data)
        fitparams, _ = fitter._model_to_fit_params(new_model)
        utils.assert_allclose(fitparams, parameters, atol=self.fit_error)

    @pytest.mark.parametrize(('model_class', 'mode'), list(itertools.product(models_2D.keys(), modes)))
    def test_pixel_sum_2D(self, model_class, mode):
        """
        Test if the sum of all pixels corresponds nearly to the integral.
        """
        parameters = models_2D[model_class]['parameters']
        model = create_model(model_class, parameters)
        values = discretize_model(model, models_2D[model_class]['x_lim'],
                                  models_2D[model_class]['y_lim'], mode=mode)
        utils.assert_allclose(values.sum(), models_2D[model_class]['integral'], rtol=0.1)


def create_model(model_class, parameters, use_constraints=True):
    """
    Create instance of model class.
    """
    constraints = {}

    if "requires_scipy" in models_2D[model_class] and not HAS_SCIPY:
        pytest.skip("SciPy not found")
    if use_constraints:
        if 'constraints' in models_2D[model_class]:
            constraints = models_2D[model_class]['constraints']
    return model_class(*parameters, **constraints)

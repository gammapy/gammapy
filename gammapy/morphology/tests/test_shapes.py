# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from astropy.tests.helper import pytest
from astropy.modeling.tests.test_models import Fittable2DModelTester
from ...morphology import Sphere2D, Shell2D, Delta2D

models_2D = {

    Sphere2D: {
        'parameters': [1, 0, 0, 10, False],
        'x_values': [0, 10, 5],
        'y_values': [0, 10, 0],
        'z_values': [1, 0, np.sqrt(75) / 10],
        'x_lim': [-11, 11],
        'y_lim': [-11, 11],
        'integral': 4. / 3 * np.pi * 10 ** 3 / (2 * 10),
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
        'parameters': [1, 0, 0, 9, 1, 10, False],
        'x_values': [0],
        'y_values': [9],
        'z_values': [1],
        'x_lim': [-11, 11],
        'y_lim': [-11, 11],
        'integral': 2 * np.pi / 3 * (10 ** 3 - 9 ** 3) / np.sqrt(10 ** 2 - 9 ** 2),
    },

    Sphere2D: {
        'parameters': [(4. / 3 * np.pi * 10 ** 3 / (2 * 10)), 0, 0, 10, True],
        'constraints': {'fixed': {'amplitude': True, 'x_0': True, 'y_0': True}},
        'x_values': [0, 10, 5],
        'y_values': [0, 10, 0],
        'z_values': [1, 0, np.sqrt(75) / 10],
        'x_lim': [-11, 11],
        'y_lim': [-11, 11],
        'integral': (4. / 3 * np.pi * 10 ** 3 / (2 * 10)),
    },

    Shell2D: {
        'parameters': [(2 * np.pi / 3 * (10 ** 3 - 8 ** 3) /
                        np.sqrt(10 ** 2 - 8 ** 2)), 0, 0, 8, 2, 10, True],
        'constraints': {'fixed': {'amplitude': True, 'x_0': True, 'y_0': True, 'width': True}},
        'x_values': [0],
        'y_values': [8],
        'z_values': [1.],
        'x_lim': [-11, 11],
        'y_lim': [-11, 11],
        'integral': (2 * np.pi / 3 * (10 ** 3 - 8 ** 3) /
                     np.sqrt(10 ** 2 - 8 ** 2)),
    }
}


@pytest.mark.parametrize(('model_class', 'test_parameters'), list(models_2D.items()))
class TestMorphologyModels(Fittable2DModelTester):
    pass

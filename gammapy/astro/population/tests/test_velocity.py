# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import pytest
from astropy.modeling.tests.test_models import Fittable1DModelTester
from ..velocity import (
    FaucherKaspi2006VelocityMaxwellian,
    Paczynski1990Velocity,
    FaucherKaspi2006VelocityBimodal,
    VMAX,
    VMIN,
)

velocity_models_1D = {
    FaucherKaspi2006VelocityMaxwellian: {
        "parameters": [1, 265],
        "x_values": [1, 10, 100, 1000],
        "y_values": [4.28745276e-08, 4.28443169e-06, 3.99282978e-04, 3.46767268e-05],
        "x_lim": [VMIN.value, VMAX.value],
        "integral": 1,
    },
    FaucherKaspi2006VelocityBimodal: {
        "parameters": [1, 160, 780, 0.9],
        "constraints": {"fixed": {"sigma_2": True, "w": True}},
        "x_values": [1, 10, 100, 1000],
        "y_values": [1.754811e-07, 1.751425e-05, 1.443781e-03, 7.391701e-05],
        "x_lim": [VMIN.value, VMAX.value],
        "integral": 1,
    },
    Paczynski1990Velocity: {
        "parameters": [1, 560],
        "x_values": [1, 10, 100, 1000],
        "y_values": [0.00227363, 0.00227219, 0.00213529, 0.00012958],
        "x_lim": [VMIN.value, VMAX.value],
        "integral": 1,
    },
}


@pytest.mark.parametrize(
    ("model_class", "test_parameters"), list(velocity_models_1D.items())
)
class TestMorphologyModels(Fittable1DModelTester):
    @classmethod
    def setup_class(cls):
        cls.N = 100
        cls.M = 100
        cls.eval_error = 0.0001
        cls.fit_error = 10
        cls.x = 5.3
        cls.y = 6.7
        cls.x1 = np.arange(1, 10, 0.1)
        cls.y1 = np.arange(1, 10, 0.1)
        cls.y2, cls.x2 = np.mgrid[:10, :8]

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pytest
from astropy.modeling.tests.test_models import Fittable1DModelTester
from ..spatial import (
    FaucherKaspi2006,
    Lorimer2006,
    YusifovKucuk2004,
    YusifovKucuk2004B,
    Paczynski1990,
    CaseBattacharya1998,
    RMIN,
    RMAX,
    ZMIN,
    ZMAX,
    Exponential,
)
from .test_velocity import velocity_models_1D

radial_models_1D = {
    FaucherKaspi2006: {
        "parameters": [1, 7.04, 1.83],
        "x_values": [0.1, 1, 10],
        "y_values": [0.00022217, 0.00127107, 0.07972058],
        "x_lim": [RMIN.value, RMAX.value],
        "integral": 1,
    },
    Lorimer2006: {
        "parameters": [1, 1.9, 5],
        "x_values": [0.1, 1, 10],
        "y_values": [0.03020158, 1.41289246, 0.56351182],
        "x_lim": [RMIN.value, RMAX.value],
        "integral": 1,
    },
    Paczynski1990: {
        "parameters": [1, 4.5],
        "x_values": [0.1, 1, 10],
        "y_values": [0.04829743, 0.03954259, 0.00535151],
        "x_lim": [RMIN.value, RMAX.value],
        "integral": 1,
    },
    YusifovKucuk2004: {
        "parameters": [1, 1.64, 4.01, 0.55],
        "x_values": [0.1, 1, 10],
        "y_values": [0.55044445, 1.5363482, 0.66157715],
        "x_lim": [RMIN.value, RMAX.value],
        "integral": 1,
    },
    YusifovKucuk2004B: {
        "parameters": [1, 4, 6.8],
        "x_values": [0.1, 1, 10],
        "y_values": [1.76840095e-08, 8.60773150e-05, 6.42641018e-04],
        "x_lim": [RMIN.value, RMAX.value],
        "integral": 1,
    },
    CaseBattacharya1998: {
        "parameters": [1, 2, 3.53],
        "x_values": [0.1, 1, 10],
        "y_values": [0.00453091, 0.31178967, 0.74237311],
        "x_lim": [RMIN.value, RMAX.value],
        "integral": 1,
    },
    Exponential: {
        "parameters": [1, 0.05],
        "x_values": [0, 0.25, 0.5],
        "y_values": [1.00000000e+00, 6.73794700e-03, 4.53999298e-05],
        "x_lim": [ZMIN.value, ZMAX.value],
        "integral": 1,
    },
}

radial_models_1D.update(velocity_models_1D)


@pytest.mark.parametrize(
    ("model_class", "test_parameters"), list(radial_models_1D.items())
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
        cls.x1 = np.arange(1, 10, .1)
        cls.y1 = np.arange(1, 10, .1)
        cls.y2, cls.x2 = np.mgrid[:10, :8]

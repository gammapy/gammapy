# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
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
        "y_values": [1.94792231e-07, 1.94415946e-05, 1.60234848e-03, 6.41602450e-10],
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

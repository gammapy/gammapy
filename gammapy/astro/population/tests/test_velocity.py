# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from gammapy.astro.population.velocity import (
    FaucherKaspi2006VelocityBimodal,
    FaucherKaspi2006VelocityMaxwellian,
    Paczynski1990Velocity,
)

test_cases = [
    {
        "class": FaucherKaspi2006VelocityMaxwellian,
        "x": [1, 10],
        "y": [4.28745276e-08, 4.28443169e-06],
    },
    {
        "class": FaucherKaspi2006VelocityBimodal,
        "x": [1, 10],
        "y": [1.754811e-07, 1.751425e-05],
    },
    {"class": Paczynski1990Velocity, "x": [1, 10], "y": [0.00227363, 0.00227219]},
]


@pytest.mark.parametrize("case", test_cases, ids=lambda _: _["class"].__name__)
def test_velocity_model(case):
    model = case["class"]()
    y = model(case["x"])
    assert_allclose(y, case["y"], rtol=1e-5)

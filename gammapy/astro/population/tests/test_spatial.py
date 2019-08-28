# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from gammapy.astro.population.spatial import (
    CaseBattacharya1998,
    Exponential,
    FaucherKaspi2006,
    Lorimer2006,
    Paczynski1990,
    YusifovKucuk2004,
    YusifovKucuk2004B,
)

test_cases = [
    {
        "class": FaucherKaspi2006,
        "x": [0.1, 1, 10],
        "y": [0.0002221728797095, 0.00127106525755, 0.0797205770877],
    },
    {
        "class": Lorimer2006,
        "x": [0.1, 1, 10],
        "y": [0.03020158, 1.41289246, 0.56351182],
    },
    {
        "class": Paczynski1990,
        "x": [0.1, 1, 10],
        "y": [0.04829743, 0.03954259, 0.00535151],
    },
    {
        "class": YusifovKucuk2004,
        "x": [0.1, 1, 10],
        "y": [0.55044445, 1.5363482, 0.66157715],
    },
    {
        "class": YusifovKucuk2004B,
        "x": [0.1, 1, 10],
        "y": [1.76840095e-08, 8.60773150e-05, 6.42641018e-04],
    },
    {
        "class": CaseBattacharya1998,
        "x": [0.1, 1, 10],
        "y": [0.00453091, 0.31178967, 0.74237311],
    },
    {
        "class": Exponential,
        "x": [0, 0.25, 0.5],
        "y": [1.00000000e00, 6.73794700e-03, 4.53999298e-05],
    },
]


@pytest.mark.parametrize("case", test_cases, ids=lambda _: _["class"].__name__)
def test_spatial_model(case):
    model = case["class"]()
    y = model(case["x"])
    assert_allclose(y, case["y"], rtol=1e-5)

# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
from astropy.time import Time
import astropy.units as u
from gammapy.maps.axes import TimeAxis
from gammapy.utils.testing import assert_allclose

@pytest.fixture
def time_intervals():
    t0 = Time("2020-03-19")
    t_min = t0 + np.linspace(0, 10, 20) * u.d
    t_max = t_min + 1 * u.h
    return {"t_min" : t_min, "t_max" : t_max}

def test_time_axis(time_intervals):
    axis = TimeAxis(time_intervals["t_min"], time_intervals["t_max"], name="time")

    assert axis.nbin == 20
    assert axis.name == "time"
    assert axis.node_type == "edges"
    assert_allclose(axis.time_delta.to_value("min"), 60)




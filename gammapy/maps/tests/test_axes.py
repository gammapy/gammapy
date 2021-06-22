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
    assert_allclose(axis.center[0].mjd, 58927.020833333336)
    assert "time" in axis.__str__()
    assert "20" in axis.__str__()


def test_incorrect_time_axis():
    tmin = np.linspace(0,10)*u.h
    tmax = np.linspace(1,11)*u.h
    with pytest.raises(TypeError):
        TimeAxis(tmin, tmax, name="time")

def test_bad_length_sort_time_axis(time_intervals):
    tmin = time_intervals["t_min"]

    tmax = time_intervals["t_max"][::-1]
    with pytest.raises(ValueError):
        TimeAxis(tmin, tmax, name="time")

    tmax = time_intervals["t_max"][:-1]
    with pytest.raises(ValueError):
        TimeAxis(tmin, tmax, name="time")


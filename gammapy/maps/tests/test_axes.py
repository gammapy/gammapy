# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
from astropy.time import Time
from astropy.table import Table
import astropy.units as u
from gammapy.maps.axes import TimeAxis
from gammapy.data import GTI
from gammapy.utils.testing import assert_allclose, requires_data, assert_time_allclose
from gammapy.utils.scripts import make_path

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
    tmax_reverse = time_intervals["t_max"][::-1]
    tmax_short = time_intervals["t_max"][:-1]

    with pytest.raises(ValueError):
        TimeAxis(tmin, tmax_reverse, name="time")

    with pytest.raises(ValueError):
        TimeAxis(tmin, tmax_short, name="time")


def test_coord_to_idx_time_axis(time_intervals):
    tmin = time_intervals["t_min"]
    tmax = time_intervals["t_max"]
    axis = TimeAxis(tmin, tmax, name="time")

    time = Time(58927.020833333336, format="mjd")
    times = axis.time_mid
    times[::2] += 1*u.h

    idx = axis.coord_to_idx(time)
    indices = axis.coord_to_idx(times)

    assert idx == 0
    assert_allclose(indices[1::2], [1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    assert_allclose(indices[::2], -1)


def test_from_table_time_axis():
    t0 = Time("2006-02-12", scale='utc')
    t_min = t0 + np.linspace(0, 10, 10)*u.d
    t_max = t_min+12*u.h
    cols = dict()
    cols["time_min"] = t_min
    cols["time_max"] = t_max
    cols["some_column"] = np.ones(10)
    table = Table(data=cols)

    axis = TimeAxis.from_table(table)

    assert axis.nbin == 10
    assert_allclose(axis.center[0].mjd, 53778.25)

@requires_data()
def test_from_gti_time_axis():
    filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
    filename = make_path(filename)
    gti = GTI.read(filename)

    axis = TimeAxis.from_gti(gti)
    expected = Time(53090.123451203704, format="mjd", scale="tt")
    assert_time_allclose(axis.time_min[0], expected)
    assert axis.nbin == 1
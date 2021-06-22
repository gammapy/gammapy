# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
from astropy.time import Time
from astropy.table import Table
import astropy.units as u
from gammapy.maps import RegionNDMap, MapAxis
from gammapy.maps.axes import TimeMapAxis
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
    axis = TimeMapAxis(time_intervals["t_min"], time_intervals["t_max"], name="time")

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
        TimeMapAxis(tmin, tmax, name="time")

def test_bad_length_sort_time_axis(time_intervals):
    tmin = time_intervals["t_min"]
    tmax_reverse = time_intervals["t_max"][::-1]
    tmax_short = time_intervals["t_max"][:-1]

    with pytest.raises(ValueError):
        TimeMapAxis(tmin, tmax_reverse, name="time")

    with pytest.raises(ValueError):
        TimeMapAxis(tmin, tmax_short, name="time")


def test_coord_to_idx_time_axis(time_intervals):
    tmin = time_intervals["t_min"]
    tmax = time_intervals["t_max"]
    axis = TimeMapAxis(tmin, tmax, name="time")

    time = Time(58927.020833333336, format="mjd")
    times = axis.time_mid
    times[::2] += 1*u.h

    idx = axis.coord_to_idx(time)
    indices = axis.coord_to_idx(times)

    assert idx == 0
    assert_allclose(indices[1::2], [1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    assert_allclose(indices[::2], -1)

def test_slice_time_axis(time_intervals):
    axis = TimeMapAxis(time_intervals["t_min"], time_intervals["t_max"], name="time")

    new_axis = axis.slice([2,6,9])
    squashed = axis.squash()

    assert new_axis.nbin == 3
    assert_allclose(squashed.time_max[0].mjd, 1)
    assert squashed.nbin == 1
    assert_allclose(squashed.time_max[0].mjd, 1)

def test_from_table_time_axis():
    t0 = Time("2006-02-12", scale='utc')
    t_min = t0 + np.linspace(0, 10, 10)*u.d
    t_max = t_min+12*u.h
    cols = dict()
    cols["time_min"] = t_min
    cols["time_max"] = t_max
    cols["some_column"] = np.ones(10)
    table = Table(data=cols)

    axis = TimeMapAxis.from_table(table)

    assert axis.nbin == 10
    assert_allclose(axis.center[0].mjd, 53778.25)

@requires_data()
def test_from_gti_time_axis():
    filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
    filename = make_path(filename)
    gti = GTI.read(filename)

    axis = TimeMapAxis.from_gti(gti)
    expected = Time(53090.123451203704, format="mjd", scale="tt")
    assert_time_allclose(axis.time_min[0], expected)
    assert axis.nbin == 1

def test_map_with_time_axis(time_intervals):
    time_axis = TimeMapAxis(time_intervals["t_min"], time_intervals["t_max"], name="time")
    energy_axis = MapAxis.from_energy_bounds(0.1,10, 2, unit="TeV")
    region_map = RegionNDMap.create(region="fk5; circle(0,0,0.1)", axes=[energy_axis, time_axis])

    assert region_map.geom.data_shape == (20, 2, 1, 1)

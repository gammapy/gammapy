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
from gammapy.utils.time import time_ref_to_dict


@pytest.fixture
def time_intervals():
    t0 = Time("2020-03-19")
    t_min = np.linspace(0, 10, 20) * u.d
    t_max = t_min + 1 * u.h
    return {"t_min": t_min, "t_max": t_max, "t_ref": t0}


@pytest.fixture
def time_interval():
    t0 = Time("2020-03-19")
    t_min = 1 * u.d
    t_max = 11 *u.d
    return {"t_min": t_min, "t_max": t_max, "t_ref": t0}


def test_time_axis(time_intervals):
    axis = TimeMapAxis(time_intervals["t_min"], time_intervals["t_max"], time_intervals["t_ref"])

    axis_copy = axis.copy()

    assert axis.nbin == 20
    assert axis.name == "time"
    assert axis.node_type == "intervals"

    assert_allclose(axis.time_delta.to_value("min"), 60)
    assert_allclose(axis.time_mid[0].mjd, 58927.020833333336)

    assert "time" in axis.__str__()
    assert "20" in axis.__str__()

    with pytest.raises(ValueError):
        axis.assert_name("bad")

    assert axis_copy == axis


def test_single_interval_time_axis(time_interval):
    axis = TimeMapAxis(
        edges_min=time_interval["t_min"],
        edges_max=time_interval["t_max"],
        reference_time=time_interval["t_ref"]
    )

    coord = Time(58933, format="mjd") + u.Quantity([1.5, 3.5, 10], unit="d")
    pix = axis.coord_to_pix(coord)

    assert axis.nbin == 1
    assert_allclose(axis.time_delta.to_value("d"), 10)
    assert_allclose(axis.time_mid[0].mjd, 58933)

    pix_min = axis.coord_to_pix(time_interval["t_min"] + 0.001 * u.s)
    assert_allclose(pix_min, -0.5)

    pix_max = axis.coord_to_pix(time_interval["t_max"] - 0.001 * u.s)
    assert_allclose(pix_max, 0.5)

    assert_allclose(pix, [0.15, 0.35, np.nan])


def test_slice_squash_time_axis(time_intervals):
    axis = TimeMapAxis(time_intervals["t_min"], time_intervals["t_max"], time_intervals["t_ref"])
    axis_squash = axis.squash()
    axis_slice = axis.slice(slice(1,5))

    assert axis_squash.nbin == 1
    assert_allclose(axis_squash.time_min[0].mjd, 58927)
    assert_allclose(axis_squash.time_delta.to_value("d"), 10.04166666)
    assert axis_slice.nbin == 4
    assert_allclose(axis_slice.time_delta.to_value("d")[0], 0.04166666666)
    assert axis_squash != axis_slice


def test_from_time_edges_time_axis():
    t0 = Time("2020-03-19")
    t_min = t0 + np.linspace(0, 10, 20) * u.d
    t_max = t_min + 1 * u.h

    axis = TimeMapAxis.from_time_edges(t_min, t_max)
    axis_h = TimeMapAxis.from_time_edges(t_min, t_max, unit='h')

    assert axis.nbin == 20
    assert axis.name == "time"
    assert_time_allclose(axis.reference_time, t0)
    assert_allclose(axis.time_delta.to_value("min"), 60)
    assert_allclose(axis.time_mid[0].mjd, 58927.020833333336)
    assert_allclose(axis_h.time_delta.to_value("h"), 1)
    assert_allclose(axis_h.time_mid[0].mjd, 58927.020833333336)
    assert axis == axis_h


def test_incorrect_time_axis():
    tmin = np.linspace(0, 10, 20) * u.h
    tmax = np.linspace(1, 11, 20) * u.h

    # incorrect reference time
    with pytest.raises(ValueError):
        TimeMapAxis(tmin, tmax, reference_time=51000 * u.d, name="time")

    # overlapping time intervals
    with pytest.raises(ValueError):
        TimeMapAxis(tmin, tmax, reference_time=Time.now(), name="time")


def test_bad_length_sort_time_axis(time_intervals):
    tref = time_intervals["t_ref"]
    tmin = time_intervals["t_min"]
    tmax_reverse = time_intervals["t_max"][::-1]
    tmax_short = time_intervals["t_max"][:-1]

    with pytest.raises(ValueError):
        TimeMapAxis(tmin, tmax_reverse, tref, name="time")

    with pytest.raises(ValueError):
        TimeMapAxis(tmin, tmax_short, tref, name="time")


def test_coord_to_idx_time_axis(time_intervals):
    tmin = time_intervals["t_min"]
    tmax = time_intervals["t_max"]
    tref = time_intervals["t_ref"]
    axis = TimeMapAxis(tmin, tmax, tref, name="time")

    time = Time(58927.020833333336, format="mjd")

    times = axis.time_mid
    times[::2] += 1 * u.h
    times = times.insert(0, tref-[1, 2] * u.yr)

    idx = axis.coord_to_idx(time)
    indices = axis.coord_to_idx(times)

    pix = axis.coord_to_pix(time)
    pixels = axis.coord_to_pix(times)

    assert idx == 0
    assert_allclose(indices[1::2], [-1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    assert_allclose(indices[::2], -1)
    assert_allclose(pix, 0, atol=1e-10)
    assert_allclose(pixels[1::2], [np.nan, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19])


def test_slice_time_axis(time_intervals):
    axis = TimeMapAxis(time_intervals["t_min"], time_intervals["t_max"], time_intervals["t_ref"])

    new_axis = axis.slice([2, 6, 9])
    squashed = axis.squash()

    assert new_axis.nbin == 3
    assert_allclose(squashed.time_max[0].mjd, 58937.041667)
    assert squashed.nbin == 1
    assert_allclose(squashed.time_max[0].mjd, 58937.041667)


def test_from_table_time_axis():
    t0 = Time("2006-02-12", scale='utc')
    t_min = np.linspace(0, 10, 10) * u.d
    t_max = t_min + 12 * u.h

    table = Table()
    table["TIME_MIN"] = t_min
    table["TIME_MAX"] = t_max
    table.meta.update(time_ref_to_dict(t0))
    table.meta["AXCOLS1"] = "TIME_MIN,TIME_MAX"

    axis = TimeMapAxis.from_table(table, format="gadf")

    assert axis.nbin == 10
    assert_allclose(axis.time_mid[0].mjd, 53778.25)


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
    time_axis = TimeMapAxis(time_intervals["t_min"], time_intervals["t_max"], time_intervals["t_ref"])
    energy_axis = MapAxis.from_energy_bounds(0.1,10, 2, unit="TeV")
    region_map = RegionNDMap.create(region="fk5; circle(0,0,0.1)", axes=[energy_axis, time_axis])

    assert region_map.geom.data_shape == (20, 2, 1, 1)

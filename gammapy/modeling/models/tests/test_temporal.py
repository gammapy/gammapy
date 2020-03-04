# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.table import Table
from astropy.time import Time
from gammapy.data.gti import GTI
from gammapy.modeling.models import (
    ConstantTemporalModel,
    LightCurveTemplateTemporalModel,
    ExpDecayTemporalModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import requires_data

# TODO: add light-curve test case from scratch
# only use the FITS one for I/O (or not at all)
@pytest.fixture(scope="session")
def light_curve():
    path = "$GAMMAPY_DATA/tests/models/light_curve/lightcrv_PKSB1222+216.fits"
    return LightCurveTemplateTemporalModel.read(path)


@requires_data()
def test_light_curve_str(light_curve):
    ss = str(light_curve)
    assert "LightCurveTemplateTemporalModel" in ss


@requires_data()
def test_light_curve_evaluate(light_curve):
    t = Time(59500, format="mjd")
    val = light_curve(t)
    assert_allclose(val, 0.015512, rtol=1e-5)

    t = Time(46300, format="mjd")
    val = light_curve(t, ext=3)
    assert_allclose(val, 0.01551196, rtol=1e-5)


@requires_data()
def test_light_curve_mean_norm_in_time_interval(light_curve):
    val = light_curve.mean_norm_in_time_interval(59530, 59600)
    assert_allclose(val, 0.030119, rtol=1e-5)


def rate(x, c="1e4 s"):
    c = u.Quantity(c)
    return np.exp(-x / c)


def ph_curve(x, amplitude=0.5, x0=0.01):
    return 100.0 + amplitude * np.sin(2 * np.pi * (x - x0) / 1.0)


def test_time_sampling(tmp_path):
    time = np.arange(0, 10, 0.06) * u.hour

    table = Table()
    table["TIME"] = time
    table["NORM"] = rate(time)
    table.meta = dict(MJDREFI=55197.0, MJDREFF=0, TIMEUNIT="hour")
    temporal_model = LightCurveTemplateTemporalModel(table)

    filename = str(make_path(tmp_path / "tmp.fits"))
    temporal_model.write(path=filename)
    model_read = temporal_model.read(filename)
    assert temporal_model.filename == filename
    assert model_read.filename == filename
    assert_allclose(model_read.table["TIME"].quantity.value, time.value)

    t_ref = "2010-01-01T00:00:00"
    t_min = "2010-01-01T00:00:00"
    t_max = "2010-01-01T08:00:00"

    sampler = temporal_model.sample_time(
        n_events=2, t_min=t_min, t_max=t_max, random_state=0, t_delta="10 min"
    )

    sampler = u.Quantity((sampler - Time(t_ref)).sec, "s")
    assert len(sampler) == 2
    assert_allclose(sampler.value, [12661.65802564, 7826.92991], rtol=1e-5)

    table = Table()
    table["TIME"] = time
    table["NORM"] = np.ones(len(time))
    table.meta = dict(MJDREFI=55197.0, MJDREFF=0, TIMEUNIT="hour")
    temporal_model_uniform = LightCurveTemplateTemporalModel(table)

    sampler_uniform = temporal_model_uniform.sample_time(
        n_events=2, t_min=t_min, t_max=t_max, random_state=0, t_delta="10 min"
    )
    sampler_uniform = u.Quantity((sampler_uniform - Time(t_ref)).sec, "s")

    assert len(sampler_uniform) == 2
    assert_allclose(sampler_uniform.value, [1261.65802564, 6026.9299098], rtol=1e-5)


def test_lightcurve_temporal_model_integral():
    time = np.arange(0, 10, 0.06) * u.hour
    table = Table()
    table["TIME"] = time
    table["NORM"] = np.ones(len(time))
    table.meta = dict(MJDREFI=55197.0, MJDREFF=0, TIMEUNIT="hour")
    temporal_model = LightCurveTemplateTemporalModel(table)

    start = [1, 3, 5] * u.hour
    stop = [2, 3.5, 6] * u.hour
    gti = GTI.create(start, stop, reference_time=Time("2010-01-01T00:00:00"))

    val = temporal_model.integral(gti.time_start, gti.time_stop)
    assert len(val) == 3
    assert_allclose(np.sum(val), 1.0, rtol=1e-5)


def test_constant_temporal_model_sample():
    temporal_model = ConstantTemporalModel()

    t_ref = "2010-01-01T00:00:00"
    t_min = "2010-01-01T00:00:00"
    t_max = "2010-01-01T08:00:00"

    sampler = temporal_model.sample_time(
        n_events=2, t_min=t_min, t_max=t_max, random_state=0
    )

    sampler = u.Quantity((sampler - Time(t_ref)).sec, "s")

    assert len(sampler) == 2
    assert_allclose(sampler.value, [15805.82891311, 20597.45375153], rtol=1e-5)


def test_constant_temporal_model_evaluate():
    temporal_model = ConstantTemporalModel()
    t = Time(46300, format="mjd")
    val = temporal_model(t)
    assert_allclose(val, 1.0, rtol=1e-5)


def test_constant_temporal_model_integral():
    temporal_model = ConstantTemporalModel()
    start = [1, 3, 5] * u.day
    stop = [2, 3.5, 6] * u.day
    gti = GTI.create(start, stop)
    val = temporal_model.integral(gti.time_start, gti.time_stop)
    assert len(val) == 3
    assert_allclose(np.sum(val), 1.0, rtol=1e-5)


def test_exponential_temporal_model_evaluate():
    temporal_model = ExpDecayTemporalModel()
    t = Time(46301, format="mjd")
    t_ref = 46300
    t0 = 2.0 * u.d
    val = temporal_model(t, t_ref=t_ref, t0=t0)
    assert_allclose(val, 0.6065306597126334, rtol=1e-5)


def test_exponential_temporal_model_integral():
    temporal_model = ExpDecayTemporalModel()
    start = [1, 3, 5] * u.day
    stop = [2, 3.5, 6] * u.day
    t_ref = Time(55555, format="mjd")
    gti = GTI.create(start, stop, reference_time=t_ref)
    val = temporal_model.integral(gti.time_start, gti.time_stop)
    assert len(val) == 3
    assert_allclose(np.sum(val), 0.1024784, rtol=1e-5)


@requires_data()
def test_to_dict(light_curve):

    out = light_curve.to_dict()
    assert out["type"] == "LightCurveTemplateTemporalModel"
    assert "lightcrv_PKSB1222+216.fits" in out["filename"]


@requires_data()
def test_with_skymodel(light_curve):

    sky_model = SkyModel(spectral_model=PowerLawSpectralModel())
    out = sky_model.to_dict()
    assert "temporal" not in out

    sky_model = SkyModel(
        spectral_model=PowerLawSpectralModel(), temporal_model=light_curve
    )
    assert sky_model.temporal_model.tag == "LightCurveTemplateTemporalModel"

    out = sky_model.to_dict()
    assert "temporal" in out

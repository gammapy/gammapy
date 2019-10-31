# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.table import Table
from astropy.time import Time
from gammapy.modeling.models import (
    ConstantTemporalModel,
    LightCurveTemplateTemporalModel,
    PhaseCurveTemplateTemporalModel,
)
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def phase_curve():
    path = make_path("$GAMMAPY_DATA/tests/phasecurve_LSI_DC.fits")
    table = Table.read(path)
    return PhaseCurveTemplateTemporalModel(
        table, time_0=43366.275, phase_0=0.0, f0=4.367575e-7
    )


@requires_data()
def test_phasecurve_phase(phase_curve):
    time = 46300.0
    phase = phase_curve.phase(time)
    assert_allclose(phase, 0.7066006737999402)


@requires_data()
def test_phasecurve_evaluate(phase_curve):
    time = 46300.0
    value = phase_curve.evaluate_norm_at_time(time)
    assert_allclose(value, 0.49059393580053845)

    phase = phase_curve.phase(time)
    value = phase_curve.evaluate_norm_at_phase(phase)
    assert_allclose(value, 0.49059393580053845)


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
def test_light_curve_evaluate_norm_at_time(light_curve):
    val = light_curve.evaluate_norm_at_time(46300)
    assert_allclose(val, 0.021192223042749835)


@requires_data()
def test_light_curve_mean_norm_in_time_interval(light_curve):
    val = light_curve.mean_norm_in_time_interval(46300, 46301)
    assert_allclose(val, 0.021192284384617066)


def rate(x, c="1e4 s"):
    c = u.Quantity(c)
    return np.exp(-x / c)


def ph_curve(x, amplitude=0.5, x0=0.01):
    return 100.0 + amplitude * np.sin(2 * np.pi * (x - x0) / 1.0)


def test_time_sampling():
    time = np.arange(0, 10, 0.06) * u.hour

    table = Table()
    table["TIME"] = time
    table["NORM"] = rate(time)
    temporal_model = LightCurveTemplateTemporalModel(table)

    t_ref = "2010-01-01T00:00:00"
    t_min = "2010-01-01T00:00:00"
    t_max = "2010-01-01T08:00:00"

    sampler = temporal_model.sample_time(
        n_events=2, t_min=t_min, t_max=t_max, random_state=0, t_delta="10 min"
    )

    sampler = u.Quantity((sampler - Time(t_ref)).sec, "s")

    table = Table()
    table["TIME"] = time
    table["NORM"] = np.ones(len(time))
    temporal_model_uniform = LightCurveTemplateTemporalModel(table)

    sampler_uniform = temporal_model_uniform.sample_time(
        n_events=2, t_min=t_min, t_max=t_max, random_state=0, t_delta="10 min"
    )

    sampler_uniform = u.Quantity((sampler_uniform - Time(t_ref)).sec, "s")

    assert len(sampler) == 2
    assert len(sampler_uniform) == 2
    assert_allclose(sampler.value, [12661.65802564, 26.9299098], rtol=1e-5)
    assert_allclose(sampler_uniform.value, [1261.65802564, 6026.9299098], rtol=1e-5)


def test_constant_temporal_model_sample():
    temporal_model = ConstantTemporalModel(norm=10.0)

    t_ref = "2010-01-01T00:00:00"
    t_min = "2010-01-01T00:00:00"
    t_max = "2010-01-01T08:00:00"

    sampler = temporal_model.sample_time(
        n_events=2, t_min=t_min, t_max=t_max, random_state=0
    )

    sampler = u.Quantity((sampler - Time(t_ref)).sec, "s")

    assert len(sampler) == 2
    assert_allclose(sampler.value, [15805.82891311, 20597.45375153], rtol=1e-5)


def test_constant_temporal_model_evaluate_norm_at_time():
    temporal_model = ConstantTemporalModel(norm=10.0)
    val = temporal_model.evaluate_norm_at_time(46300)
    assert_allclose(val, 10.0, rtol=1e-5)


def test_phase_time_sampling():
    time_0 = "2010-01-01T00:00:00"
    phase = np.arange(0, 1, 0.01)

    table = Table()
    table["PHASE"] = phase
    table["NORM"] = ph_curve(phase)
    phase_model = PhaseCurveTemplateTemporalModel(
        table, time_0=Time(time_0).mjd, phase_0=0.0, f0=2, f1=0.0, f2=0.0
    )

    t_min = "2010-01-01T00:00:00"
    t_max = "2010-01-01T08:00:00"

    sampler = phase_model.sample_time(
        n_events=2, t_min=t_min, t_max=t_max, random_state=0, t_delta="0.01 s"
    )

    sampler = u.Quantity((sampler - Time(time_0)).sec, "s")

    assert len(sampler) == 2
    assert_allclose(sampler.value, [8525.00102763, 11362.44044883], rtol=1e-5)

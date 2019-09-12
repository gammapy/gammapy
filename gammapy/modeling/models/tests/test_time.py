# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.table import Table
from astropy.time import Time
from gammapy.modeling.models import LightCurveTableModel, PhaseCurveTableModel
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def phase_curve():
    filename = make_path("$GAMMAPY_DATA/tests/phasecurve_LSI_DC.fits")
    table = Table.read(str(filename))
    return PhaseCurveTableModel(
        table, time_0=43366.275, phase_0=0.0, f0=4.367575e-7, f1=0.0, f2=0.0
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
    return LightCurveTableModel.read(path)


@requires_data()
def test_light_curve_str(light_curve):
    ss = str(light_curve)
    assert "LightCurveTableModel" in ss


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


def test_map_sampling():
    time = np.arange(0, 10, 0.06) * u.hour

    table = Table()
    table["TIME"] = time
    table["NORM"] = rate(time)
    temporal_model = LightCurveTableModel(table)

    t_min = "2010-01-01T00:00:00"
    t_max = "2010-01-01T08:00:00"

    sampler = temporal_model.sample_time(
        n_events=2, t_min=t_min, t_max=t_max, random_state=0, t_delta="10 min"
    )

    sampler_no_lc = LightCurveTableModel(None).sample_time(
        n_events=2, t_min=t_min, t_max=t_max, random_state=0, t_delta="10 min"
    )

    assert len(sampler) == 2
    assert len(sampler_no_lc) == 2
    assert_allclose(sampler.value, [12661.65802564, 26.9299098], rtol=1e-5)
    assert_allclose(sampler_no_lc.value, [15805.82891311, 20597.45375153], rtol=1e-5)

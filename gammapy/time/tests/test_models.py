# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose
from astropy.table import Table
from ...utils.scripts import make_path
from ...utils.testing import requires_data, requires_dependency
from ..models import PhaseCurveTableModel, LightCurveTableModel


@pytest.fixture(scope="session")
def phase_curve():
    filename = make_path("$GAMMAPY_EXTRA/test_datasets/phasecurve_LSI_DC.fits")
    table = Table.read(str(filename))
    return PhaseCurveTableModel(
        table, time_0=43366.275, phase_0=0.0, f0=4.367575e-7, f1=0.0, f2=0.0
    )


@requires_data("gammapy-extra")
def test_phasecurve_phase(phase_curve):
    time = 46300.0
    phase = phase_curve.phase(time)
    assert_allclose(phase, 0.7066006737999402)


@requires_data("gammapy-extra")
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
    path = "$GAMMAPY_EXTRA/test_datasets/models/light_curve/lightcrv_PKSB1222+216.fits"
    return LightCurveTableModel.read(path)


@requires_data("gammapy-extra")
def test_light_curve_str(light_curve):
    ss = str(light_curve)
    assert "LightCurveTableModel" in ss


@requires_dependency("scipy")
@requires_data("gammapy-extra")
def test_light_curve_evaluate_norm_at_time(light_curve):
    val = light_curve.evaluate_norm_at_time(46300)
    assert_allclose(val, 0.021192223042749835)


@requires_dependency("scipy")
@requires_data("gammapy-extra")
def test_light_curve_mean_norm_in_time_interval(light_curve):
    val = light_curve.mean_norm_in_time_interval(46300, 46301)
    assert_allclose(val, 0.021192284384617066)

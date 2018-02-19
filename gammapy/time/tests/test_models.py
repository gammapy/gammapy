# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.table import Table
import pytest
from numpy.testing import assert_allclose
from ...utils.scripts import make_path
from ...utils.testing import requires_data
from ..models import PhaseCurve


@pytest.fixture(scope='session')
def phase_curve():
    filename = make_path('$GAMMAPY_EXTRA/test_datasets/phasecurve_LSI_DC.fits')
    table = Table.read(str(filename))
    return PhaseCurve(table, time_0=43366.275, phase_0=0.0, f0=4.367575e-7, f1=0.0, f2=0.0)


@requires_data('gammapy-extra')
def test_phasecurve_phase(phase_curve):
    time = 46300.0
    phase = phase_curve.phase(time)
    assert_allclose(phase, 0.7066006737999402)


@requires_data('gammapy-extra')
def test_phasecurve_evaluate(phase_curve):
    time = 46300.0
    value = phase_curve.evaluate_norm_at_time(time)
    assert_allclose(value, 0.49059393580053845)

    phase = phase_curve.phase(time)
    value = phase_curve.evaluate_norm_at_phase(phase)
    assert_allclose(value, 0.49059393580053845)

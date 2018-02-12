from astropy.table import Table
import numpy as np
import pytest
from numpy.testing import assert_allclose
from gammapy.time.models import PhaseCurve
from gammapy.utils.testing import requires_data
from gammapy.utils.scripts import make_path



@pytest.fixture(scope='session')
def phase_curve():
    filename = make_path('$GAMMAPY_EXTRA/test_datasets/phasecurve_LSI_DC.fits')
    print(filename)
    table = Table.read(str(filename))
    return PhaseCurve(table,reference=43366.275,phase0 = 0.0,f0 = 0.5,f1 = 0.0,f2 = 0.0
)

# To check the value of the phase
def test_phasecurve_phase(phase_curve):
    time = 46300.0
    assert_allclose(phase_curve.phase(time),0.86,rtol=0.01)

# To check the value of the normalization constant
def test_phasecurve_norm(phase_curve):
    time = 46300.0
    assert_allclose(phase_curve.evaluate(time),0.05,rtol=0.01)

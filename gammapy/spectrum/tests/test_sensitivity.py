# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.utils.testing import requires_data, requires_dependency
from ...irf.io import CTAPerf
from ..sensitivity import SensitivityEstimator


@pytest.fixture()
def sens():
    filename = "$GAMMAPY_EXTRA/datasets/cta/perf_prod2/point_like_non_smoothed/South_5h.fits.gz"
    irf = CTAPerf.read(filename)
    sens = SensitivityEstimator(irf=irf, livetime=5.0 * u.h)
    sens.run()
    return sens


@requires_dependency("scipy")
@requires_data("gammapy-extra")
def test_cta_sensitivity_estimator(sens):
    table = sens.results_table

    assert len(table) == 21
    assert table.colnames == ["energy", "e2dnde", "excess", "background", "criterion"]
    assert table["energy"].unit == "TeV"
    assert table["e2dnde"].unit == "erg / (cm2 s)"

    row = table[0]
    assert_allclose(row["energy"], 0.015848, rtol=1e-3)
    assert_allclose(row["e2dnde"], 1.2656e-10, rtol=1e-3)
    assert_allclose(row["excess"], 339.143, rtol=1e-3)
    assert_allclose(row["background"], 3703.48, rtol=1e-3)
    assert row["criterion"] == "significance"

    row = table[9]
    assert_allclose(row["energy"], 1, rtol=1e-3)
    assert_allclose(row["e2dnde"], 4.28759e-13, rtol=1e-3)
    assert_allclose(row["excess"], 18.1072, rtol=1e-3)
    assert_allclose(row["background"], 5.11857, rtol=1e-3)
    assert row["criterion"] == "significance"

    row = table[20]
    assert_allclose(row["energy"], 158.489, rtol=1e-3)
    assert_allclose(row["e2dnde"], 9.0483e-12, rtol=1e-3)
    assert_allclose(row["excess"], 10, rtol=1e-3)
    assert_allclose(row["background"], 0.00566093, rtol=1e-3)
    assert row["criterion"] == "gamma"

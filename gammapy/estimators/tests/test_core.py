# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.estimators import ESTIMATOR_REGISTRY


def test_estimator_registry():
    assert "Estimator" in str(ESTIMATOR_REGISTRY)

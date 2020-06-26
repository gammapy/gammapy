from gammapy.estimators import ESTIMATOR_REGISTRY


def test_estimator_registry():
    assert "Estimator" in str(ESTIMATOR_REGISTRY)

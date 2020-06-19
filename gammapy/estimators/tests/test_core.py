from gammapy.estimators import ESTIMATORS


def test_estimator_registry():
    assert "Estimator" in str(ESTIMATORS)

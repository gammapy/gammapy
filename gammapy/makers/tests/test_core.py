from gammapy.makers import MAKER_REGISTRY


def test_maker_registry():
    assert "Maker" in str(MAKER_REGISTRY)

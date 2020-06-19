from gammapy.makers import MAKERS


def test_maker_registry():
    assert "Maker" in str(MAKERS)

from ..analysis import Analysis


def test_config():
    assert Analysis().validate_schema() is None

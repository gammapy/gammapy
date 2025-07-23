# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings
import pytest
import yaml
from numpy.testing import assert_allclose
from astropy.table import Table
from gammapy.utils.scripts import (
    get_images_paths,
    make_path,
    read_yaml,
    recursive_merge_dicts,
    requires_module,
    to_yaml,
    write_yaml,
    logic_parser,
)


@pytest.mark.xfail
def test_get_images_paths():
    assert any("images" in str(p) for p in get_images_paths())


def test_recursive_merge_dicts():
    old = dict(a=42, b=dict(c=43, e=44))
    update = dict(d=99, b=dict(g=98, c=50))

    new = recursive_merge_dicts(old, update)
    assert new["b"]["c"] == 50
    assert new["b"]["e"] == 44
    assert new["b"]["g"] == 98
    assert new["a"] == 42
    assert new["d"] == 99


def test_read_write_yaml_checksum(tmp_path):
    data = to_yaml({"b": 1234, "a": "other"})
    path = make_path(tmp_path / "test.yaml")
    write_yaml(data, path, sort_keys=False, checksum=True)

    yaml_content = path.read_text()
    assert "checksum: " in yaml_content

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = read_yaml(path, checksum=True)
    assert "checksum" not in res.keys()

    yaml_content = yaml_content.replace("1234", "12345")
    bad = make_path(tmp_path) / "bad_checksum.yaml"
    bad.write_text(yaml_content)
    with pytest.warns(UserWarning):
        read_yaml(bad, checksum=True)

    res["checksum"] = "bad"
    yaml_str = yaml.dump(data, sort_keys=True, default_flow_style=False)
    path.write_text(yaml_str)

    with pytest.warns(UserWarning):
        read_yaml(bad, checksum=True)


def test_requires_module():
    class MyClass:
        @requires_module("math")
        def method(self):
            import math

            return math.sqrt(9)

        @requires_module("nonexistent_module")
        def method_unavailable(self):
            return "Should not be called"

        @requires_module("math")
        @property
        def prop(self):
            import math

            return math.sqrt(16)

        @requires_module("nonexistent_module")
        @property
        def prop_unavailable(self):
            return "Should not be called"

    result = MyClass()

    assert_allclose(result.method(), 3.0)

    with pytest.raises(
        ImportError,
        match="The 'nonexistent_module' module is required to use this method.",
    ):
        result.method_unavailable()

    assert_allclose(result.prop, 4.0)

    with pytest.raises(
        ImportError,
        match="The 'nonexistent_module' module is required to use this property.",
    ):
        _ = result.prop_unavailable

    with pytest.raises(
        TypeError, match="requires_module can only be used on methods or properties."
    ):

        @requires_module("nonexistent_module")
        class InvalidUsage:
            pass


def test_logic_parser():
    data = {"OBS_ID": [1, 2, 3, 4], "EVENT_TYPE": ["1", "3", "4", "2"]}
    table = Table(data)

    # Test 'and' operation
    result = logic_parser(table, "(OBS_ID < 3) and (OBS_ID > 1)")
    assert len(result) == 1
    assert result["OBS_ID"][0] == 2

    # Test 'or' operation
    result = logic_parser(table, "(OBS_ID < 2) or (OBS_ID > 3)")
    assert len(result) == 2
    assert result["OBS_ID"][0] == 1
    assert result["OBS_ID"][1] == 4

    # Test '==' operation
    result = logic_parser(table, 'EVENT_TYPE == "3"')
    assert len(result) == 1
    assert result["EVENT_TYPE"][0] == "3"

    # Test '!=' operation
    result = logic_parser(table, 'EVENT_TYPE != "3"')
    assert len(result) == 3
    assert result["EVENT_TYPE"][0] == "1"
    assert result["EVENT_TYPE"][1] == "4"
    assert result["EVENT_TYPE"][2] == "2"

    # Test '<' operation
    result = logic_parser(table, "OBS_ID < 3")
    assert len(result) == 2
    assert result["OBS_ID"][0] == 1
    assert result["OBS_ID"][1] == 2

    # Test '<=' operation
    result = logic_parser(table, "OBS_ID <= 3")
    assert len(result) == 3
    assert result["OBS_ID"][0] == 1
    assert result["OBS_ID"][1] == 2
    assert result["OBS_ID"][2] == 3

    # Test '>' operation
    result = logic_parser(table, "OBS_ID > 2")
    assert len(result) == 2
    assert result["OBS_ID"][0] == 3
    assert result["OBS_ID"][1] == 4

    # Test '>=' operation
    result = logic_parser(table, "OBS_ID >= 2")
    assert len(result) == 3
    assert result["OBS_ID"][0] == 2
    assert result["OBS_ID"][1] == 3
    assert result["OBS_ID"][2] == 4

    # Test 'in' operation
    result = logic_parser(table, 'EVENT_TYPE in ["3", "4"]')
    assert len(result) == 2
    assert result["EVENT_TYPE"][0] == "3"
    assert result["EVENT_TYPE"][1] == "4"

    # Test 'not in' operation
    result = logic_parser(table, 'EVENT_TYPE not in ["3", "4"]')
    assert len(result) == 2
    assert result["EVENT_TYPE"][0] == "1"
    assert result["EVENT_TYPE"][1] == "2"

    # Test no match
    result = logic_parser(table, "(OBS_ID < 3) and (OBS_ID > 2)")
    assert len(result) == 0


def test_logic_parser_error():
    # Create a sample table
    data = {"OBS_ID": [1, 2, 3, 4], "EVENT_TYPE": ["1", "3", "4", "2"]}
    table = Table(data)

    # Define an expression with a non-existent key
    expression = "(NON_EXISTENT_KEY < 3) and (OBS_ID > 1)"

    with pytest.raises(KeyError) as excinfo:
        logic_parser(table, expression)

    with pytest.raises(ValueError) as excinfo:
        expression = "unsupported_expression()"
        logic_parser(table, expression)
    assert "Unsupported expression type" in str(excinfo.value)

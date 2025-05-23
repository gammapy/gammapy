# Licensed under a 3-clause BSD style license - see LICENSE.rst
import yaml
from gammapy.utils.check import add_checksum, verify_checksum, yaml_checksum


def test_yaml_checksum():
    data = {
        "a": 50,
        "b": 3.14e-12,
    }
    yaml_str = yaml.dump(
        data, sort_keys=False, indent=4, width=80, default_flow_style=False
    )
    checksum = yaml_checksum(yaml_str)

    assert checksum == "47fd166725c49519c7c31c19f53b53dd"
    assert verify_checksum(yaml_str, "47fd166725c49519c7c31c19f53b53dd")

    yaml_with_checksum = add_checksum(yaml_str)
    assert "checksum: 47fd166725c49519c7c31c19f53b53dd" in yaml_with_checksum

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from gammapy.utils.scripts import get_images_paths, recursive_merge_dicts


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

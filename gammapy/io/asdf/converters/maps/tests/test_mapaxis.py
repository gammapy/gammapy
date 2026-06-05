# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import pytest

from gammapy.maps import MapAxis

asdf = pytest.importorskip("asdf")
pytest.importorskip("asdf.testing")
from asdf.testing.helpers import yaml_to_asdf  # noqa: E402


tested_map_axes = [
    MapAxis(nodes=[0.25, 0.75, 1.0, 2.0], interp="lin", node_type="edges"),
    MapAxis(
        nodes=[1, 3, 7], unit="TeV", name="energy", interp="log", node_type="center"
    ),
    MapAxis(
        nodes=[0.25, 0.75, 1.0, 2.0],
        unit="TeV",
        name="energy",
        interp="sqrt",
        node_type="edges",
    ),
    MapAxis(nodes=np.array([0.25, 0.75, 1.0, 2.0]), interp="lin", node_type="center"),
    MapAxis(nodes=[0, 1, 3, 7], unit="deg", interp="lin", boundary_type="periodic"),
]


@pytest.mark.parametrize("axis", tested_map_axes)
def test_map_axis_roundtrip(axis, tmp_path):
    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["axis"] = axis
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        assert af["axis"] == axis


tested_read_examples = [
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/mapaxis-1.0.0>
          name: energy
          interp: log
          node_type: edges
          boundary_type: monotonic
          nodes: !core/ndarray-1.1.0 [0.25, 0.75, 1.0, 2.0]
          unit: TeV""",
        "truth": MapAxis(
            nodes=[0.25, 0.75, 1.0, 2.0],
            name="energy",
            unit="TeV",
            interp="log",
            node_type="edges",
            boundary_type="monotonic",
        ),
    },
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/mapaxis-1.0.0>
          name: energy
          interp: log
          node_type: edges
          nodes: !core/ndarray-1.1.0 [0.25, 0.75, 1.0, 2.0]
          unit: TeV""",
        "truth": MapAxis(
            nodes=[0.25, 0.75, 1.0, 2.0],
            name="energy",
            unit="TeV",
            interp="log",
            node_type="edges",
            boundary_type="monotonic",
        ),
    },
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/mapaxis-1.0.0>
          name: energy
          interp: bad
          node_type: edges
          nodes: !core/ndarray-1.1.0 [0.25, 0.75, 1.0, 2.0]
          unit: TeV""",
    },
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/mapaxis-1.0.0>
              name: energy
              interp: lin
              node_type: edges""",
    },
]


@pytest.mark.parametrize("example", tested_read_examples)
def test_map_axis_read_examples(example):
    buff = yaml_to_asdf(f"example: {example['example'].strip()}")

    if example.get("truth") is not None:
        with asdf.open(buff) as af:
            assert af["example"] == example["truth"]
    else:
        with pytest.raises(asdf.exceptions.ValidationError):
            asdf.open(buff)

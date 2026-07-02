# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import pytest

from astropy import units as u
from astropy.time import Time
from gammapy.maps import LabelMapAxis, MapAxes, MapAxis, TimeMapAxis
from gammapy.utils.testing import assert_time_allclose

asdf = pytest.importorskip("asdf")
pytest.importorskip("asdf.testing")
from asdf.testing.helpers import yaml_to_asdf  # noqa: E402


tested_map_axis = [
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


@pytest.mark.parametrize("axis", tested_map_axis)
def test_map_axis_roundtrip(axis, tmp_path):
    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["axis"] = axis
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        assert af["axis"] == axis


tested_read_mapaxis_examples = [
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


@pytest.mark.parametrize("example", tested_read_mapaxis_examples)
def test_map_axis_read_examples(example):
    buff = yaml_to_asdf(f"example: {example['example'].strip()}")

    if example.get("truth") is not None:
        with asdf.open(buff) as af:
            assert af["example"] == example["truth"]
    else:
        with pytest.raises(asdf.exceptions.ValidationError):
            asdf.open(buff)


tested_time_map_axis = [
    TimeMapAxis(
        edges_min=[1] * u.d, edges_max=[11] * u.d, reference_time=Time("2020-03-19")
    ),
    TimeMapAxis(
        edges_min=[0, 1, 3] * u.d,
        edges_max=[0.8, 1.9, 5.4] * u.d,
        reference_time=Time("1999-01-01"),
    ),
    TimeMapAxis(
        edges_min=[0, 24, 48] * u.h,
        edges_max=[12, 36, 60] * u.h,
        reference_time=Time("2020-01-01"),
        name="test_time",
    ),
]


@pytest.mark.parametrize("time_axis", tested_time_map_axis)
def test_time_map_axis_roundtrip(time_axis, tmp_path):
    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["time_axis"] = time_axis
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        result = af["time_axis"]
        assert result == time_axis
        assert_time_allclose(result.reference_time, time_axis.reference_time)


tested_read_timemapaxis_examples = [
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/timemapaxis-1.0.0>
        name : time
        interp : lin
        reference_time : !time/time-1.4.0 1999-01-01 00:00:00.000
        edges_max : !unit/quantity-1.3.0
          unit: !unit/unit-1.0.0 d
          value: !core/ndarray-1.1.0
            data: [0.8, 1.9, 5.4]
        edges_min : !unit/quantity-1.3.0
           unit: !unit/unit-1.0.0 d
           value: !core/ndarray-1.1.0
             data: [0, 1, 3]
          """,
        "truth": TimeMapAxis(
            edges_min=[0, 1, 3] * u.d,
            edges_max=[0.8, 1.9, 5.4] * u.d,
            reference_time=Time("1999-01-01"),
        ),
    },
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/timemapaxis-1.0.0>
         name : test_time
         interp : lin
         reference_time : !time/time-1.4.0 2020-01-01 00:00:00.000
         edges_max : !unit/quantity-1.3.0
           unit: !unit/unit-1.0.0 h
           value: !core/ndarray-1.1.0
              data : [12, 36, 60]
         edges_min : !unit/quantity-1.3.0
            unit: !unit/unit-1.0.0 h
            value: !core/ndarray-1.1.0
              data: [0, 24, 48]
          """,
        "truth": TimeMapAxis(
            name="test_time",
            edges_min=[0, 24, 48] * u.h,
            edges_max=[12, 36, 60] * u.h,
            reference_time=Time("2020-01-01"),
        ),
    },
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/timemapaxis-1.0.0>
         name : time
         interp : lin
         reference_time : !time/time-1.4.0 2020-01-01 00:00:00.000
         edges_min : !unit/quantity-1.3.0
            unit: !unit/unit-1.0.0 d
            value: !core/ndarray-1.1.0
              data: [1]
           """,
    },
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/timemapaxis-1.0.0>
         name : time
         interp : lin
         edges_max : !unit/quantity-1.3.0
            unit: !unit/unit-1.0.0 d
            value: !core/ndarray-1.1.0
               data: [11]
         edges_min : !unit/quantity-1.3.0
            unit: !unit/unit-1.0.0 d
            value: !core/ndarray-1.1.0
              data: [1]
           """,
    },
]


@pytest.mark.parametrize("example", tested_read_timemapaxis_examples)
def test_time_map_axis_read_examples(example):
    buff = yaml_to_asdf(f"example: {example['example'].strip()}")

    if example.get("truth") is not None:
        with asdf.open(buff) as af:
            assert af["example"] == example["truth"]
    else:
        with pytest.raises(asdf.exceptions.ValidationError):
            asdf.open(buff)


def test_time_map_axis_invalid_interp():
    example = """!<asdf://gammapy.org/gammapy/tags/maps/timemapaxis-1.0.0>
             name : time
             interp : log
             reference_time : !time/time-1.4.0 2020-01-01 00:00:00.000
             edges_max : !unit/quantity-1.3.0
               unit: !unit/unit-1.0.0 d
               value: !core/ndarray-1.1.0
                 data: [11]
             edges_min : !unit/quantity-1.3.0
                unit: !unit/unit-1.0.0 d
                value: !core/ndarray-1.1.0
                 data: [1]
               """
    buff = yaml_to_asdf(f"example: {example.strip()}")
    with pytest.raises(asdf.exceptions.ValidationError):
        asdf.open(buff)


tested_label_map_axis = [
    LabelMapAxis(labels=["label-1", "label-2", "label-3"], name="label-axis"),
    LabelMapAxis(labels=["label-1", "label-2", "label-3"]),
]


@pytest.mark.parametrize("label_axis", tested_label_map_axis)
def test_label_map_axis_roundtrip(label_axis, tmp_path):
    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["label_axis"] = label_axis
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        assert af["label_axis"] == label_axis


tested_read_labelmapaxis_examples = [
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/labelmapaxis-1.0.0>
         labels: [label-1, label-2, label-3]
         name: label-axis
         """,
        "truth": LabelMapAxis(
            labels=["label-1", "label-2", "label-3"], name="label-axis"
        ),
    },
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/labelmapaxis-1.0.0>
         labels: [label-1, label-2, label-3]
         name: ""
         """,
        "truth": LabelMapAxis(labels=["label-1", "label-2", "label-3"]),
    },
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/labelmapaxis-1.0.0>
         name: ''
         """,
    },
]


@pytest.mark.parametrize("example", tested_read_labelmapaxis_examples)
def test_label_map_axis_read_examples(example):
    buff = yaml_to_asdf(f"example: {example['example'].strip()}")

    if example.get("truth") is not None:
        with asdf.open(buff) as af:
            assert af["example"] == example["truth"]
    else:
        with pytest.raises(asdf.exceptions.ValidationError):
            asdf.open(buff)


tested_map_axes = [
    MapAxes(
        [
            TimeMapAxis(
                edges_min=[0, 1, 2] * u.d,
                edges_max=[1, 2, 3] * u.d,
                reference_time=Time("1999-01-01"),
            ),
            MapAxis(nodes=[0.25, 0.75, 1.0, 2.0], interp="lin", node_type="edges"),
            LabelMapAxis(labels=["label-1", "label-2", "label-3"], name="label-axis"),
        ]
    ),
    MapAxes(
        [
            MapAxis(
                nodes=[1, 3, 7],
                unit="TeV",
                name="energy",
                interp="log",
                node_type="center",
            ),
            LabelMapAxis(labels=["label-1", "label-2", "label-3"]),
        ]
    ),
    MapAxes(
        [
            TimeMapAxis(
                edges_min=[1, 10] * u.d,
                edges_max=[2, 13] * u.d,
                reference_time=Time("2020-03-19"),
            ),
            MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=4),
        ]
    ),
    MapAxes([]),
]


@pytest.mark.parametrize("axes", tested_map_axes)
def test_map_axes_roundtrip(axes, tmp_path):
    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["axes"] = axes
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        assert af["axes"] == axes


tested_read_mapaxes_examples = [
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/mapaxes-1.0.0>
       axes:
         - !<asdf://gammapy.org/gammapy/tags/maps/timemapaxis-1.0.0>
            name : test_time
            interp : lin
            reference_time : !time/time-1.4.0 2020-01-01 00:00:00.000
            edges_max : !unit/quantity-1.3.0
               unit: !unit/unit-1.0.0 h
               value: !core/ndarray-1.1.0
                 data : [12, 36, 60]
            edges_min : !unit/quantity-1.3.0
               unit: !unit/unit-1.0.0 h
               value: !core/ndarray-1.1.0
                 data: [0, 24, 48]
         -  !<asdf://gammapy.org/gammapy/tags/maps/mapaxis-1.0.0>
              name: energy
              interp: log
              node_type: edges
              boundary_type: monotonic
              nodes: !core/ndarray-1.1.0 [0.25, 0.75, 1.0, 2.0]
              unit: TeV
         -  !<asdf://gammapy.org/gammapy/tags/maps/labelmapaxis-1.0.0>
             labels: [label-1, label-2, label-3]
             name: label-axis
             """,
        "truth": MapAxes(
            [
                TimeMapAxis(
                    name="test_time",
                    edges_min=[0, 24, 48] * u.h,
                    edges_max=[12, 36, 60] * u.h,
                    reference_time=Time("2020-01-01"),
                ),
                MapAxis(
                    nodes=[0.25, 0.75, 1.0, 2.0],
                    name="energy",
                    unit="TeV",
                    interp="log",
                    node_type="edges",
                    boundary_type="monotonic",
                ),
                LabelMapAxis(
                    labels=["label-1", "label-2", "label-3"], name="label-axis"
                ),
            ],
        ),
    },
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/mapaxes-1.0.0>""",
    },
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/mapaxes-1.0.0>
         axes:
           name: bad_item
           value: 5
        """,
    },
]


@pytest.mark.parametrize("example", tested_read_mapaxes_examples)
def test_map_axes_read_examples(example):
    buff = yaml_to_asdf(f"example: {example['example'].strip()}")

    if example.get("truth") is not None:
        with asdf.open(buff) as af:
            assert af["example"] == example["truth"]
    else:
        with pytest.raises(asdf.exceptions.ValidationError):
            asdf.open(buff)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from gammapy.irf import EDispMap, EDispKernelMap
from gammapy.maps import MapAxis

asdf = pytest.importorskip("asdf")
pytest.importorskip("asdf.testing")


def test_edispmap_roundtrip(tmp_path):
    file_path = tmp_path / "test.asdf"
    energy_axis_true = MapAxis.from_energy_bounds(
        "0.3 TeV", "10 TeV", nbin=5, name="energy_true"
    )
    edispmap = EDispMap.from_diagonal_response(energy_axis_true)

    with asdf.AsdfFile() as af:
        af["edispmap"] = edispmap
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        result = af["edispmap"]

        assert type(result) is EDispMap
        assert result.required_axes == ["migra", "energy_true"]

        assert_allclose(result.edisp_map.data, edispmap.edisp_map.data)
        assert result.edisp_map.geom == edispmap.edisp_map.geom
        assert result.edisp_map.unit == edispmap.edisp_map.unit

        assert_allclose(result.exposure_map.data, edispmap.exposure_map.data)
        assert result.exposure_map.geom == edispmap.exposure_map.geom
        assert result.exposure_map.unit == edispmap.exposure_map.unit


def test_edispmap_roundtrip_no_exposure(tmp_path):
    file_path = tmp_path / "test.asdf"
    energy_axis_true = MapAxis.from_energy_bounds(
        "0.3 TeV", "10 TeV", nbin=5, name="energy_true"
    )
    edispmap = EDispMap.from_diagonal_response(energy_axis_true)
    edispmap.exposure_map = None

    with asdf.AsdfFile() as af:
        af["edispmap"] = edispmap
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        result = af["edispmap"]

        assert type(result) is EDispMap
        assert result.required_axes == ["migra", "energy_true"]

        assert result.exposure_map is None
        assert_allclose(result.edisp_map.data, edispmap.edisp_map.data)
        assert result.edisp_map.geom == edispmap.edisp_map.geom
        assert result.edisp_map.unit == edispmap.edisp_map.unit


def test_edispkernelmap_roundtrip(tmp_path):
    file_path = tmp_path / "test.asdf"

    energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=5, name="energy")
    energy_axis_true = MapAxis.from_energy_bounds(
        "0.3 TeV", "30 TeV", nbin=10, per_decade=True, name="energy_true"
    )
    edispkernelmap = EDispKernelMap.from_diagonal_response(
        energy_axis=energy_axis, energy_axis_true=energy_axis_true
    )

    with asdf.AsdfFile() as af:
        af["edispkernelmap"] = edispkernelmap
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        result = af["edispkernelmap"]
        assert type(result) is EDispKernelMap
        assert result.required_axes == ["energy", "energy_true"]

        assert_allclose(result.edisp_map.data, edispkernelmap.edisp_map.data)
        assert result.edisp_map.geom == edispkernelmap.edisp_map.geom
        assert result.edisp_map.unit == edispkernelmap.edisp_map.unit

        assert_allclose(result.exposure_map.data, edispkernelmap.exposure_map.data)
        assert result.exposure_map.geom == edispkernelmap.exposure_map.geom
        assert result.exposure_map.unit == edispkernelmap.exposure_map.unit

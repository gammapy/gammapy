# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest

from numpy.testing import assert_allclose

from gammapy.irf import PSFMap, RecoPSFMap
from gammapy.maps import MapAxis, RegionGeom, WcsGeom
import astropy.units as u

asdf = pytest.importorskip("asdf")
pytest.importorskip("asdf.testing")


def test_psfmap_roundtrip(tmp_path):
    file_path = tmp_path / "test.asdf"
    energy_axis_true = MapAxis.from_energy_bounds(
        "1 TeV", "10 TeV", nbin=3, name="energy_true"
    )
    geom = RegionGeom.create("icrs;circle(0, 0, 0.1)")

    psf = PSFMap.from_gauss(
        energy_axis_true=energy_axis_true, sigma=[0.1, 0.2, 0.3] * u.deg, geom=geom
    )
    with asdf.AsdfFile() as af:
        af["psf"] = psf
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        result = af["psf"]

        assert type(result) is PSFMap
        assert result.required_axes == ["rad", "energy_true"]

        assert_allclose(result.psf_map.data, psf.psf_map.data)
        assert result.psf_map.geom == psf.psf_map.geom
        assert result.psf_map.unit == psf.psf_map.unit

        assert_allclose(result.exposure_map.data, psf.exposure_map.data)
        assert result.exposure_map.geom == psf.exposure_map.geom
        assert result.exposure_map.unit == psf.exposure_map.unit


def test_psfmap_roundtrip_no_exposure(tmp_path):
    file_path = tmp_path / "test.asdf"
    energy_axis_true = MapAxis.from_energy_bounds(
        "1 TeV", "10 TeV", nbin=3, name="energy_true"
    )
    geom = WcsGeom.create(
        skydir=(0, 0), frame="galactic", npix=(3, 3), binsz=20.0 * u.deg
    )
    psf = PSFMap.from_gauss(
        energy_axis_true=energy_axis_true, sigma=[0.1, 0.2, 0.3] * u.deg, geom=geom
    )
    psf.exposure_map = None

    with asdf.AsdfFile() as af:
        af["psf"] = psf
        af.write_to(file_path)
    with asdf.open(file_path) as af:
        result = af["psf"]
        assert type(result) is PSFMap
        assert result.required_axes == ["rad", "energy_true"]

        assert_allclose(result.psf_map.data, psf.psf_map.data)
        assert result.psf_map.geom == psf.psf_map.geom
        assert result.psf_map.unit == psf.psf_map.unit

        assert result.exposure_map is None


def test_recopsfmap_roundtrip(tmp_path):
    file_path = tmp_path / "test.asdf"
    energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=3, name="energy")
    geom = WcsGeom.create(
        skydir=(0, 0), frame="galactic", npix=(3, 3), binsz=20.0 * u.deg
    )
    reco_psf = RecoPSFMap.from_gauss(
        energy_axis=energy_axis, sigma=[0.1, 0.2, 0.3] * u.deg, geom=geom
    )

    with asdf.AsdfFile() as af:
        af["reco_psf"] = reco_psf
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        result = af["reco_psf"]

        assert type(result) is RecoPSFMap
        assert result.required_axes == ["rad", "energy"]

        assert_allclose(result.psf_map.data, reco_psf.psf_map.data)
        assert result.psf_map.geom == reco_psf.psf_map.geom
        assert result.psf_map.unit == reco_psf.psf_map.unit

        assert_allclose(result.exposure_map.data, reco_psf.exposure_map.data)
        assert result.exposure_map.geom == reco_psf.exposure_map.geom
        assert result.exposure_map.unit == reco_psf.exposure_map.unit

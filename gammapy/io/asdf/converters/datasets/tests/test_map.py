# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from numpy.testing import assert_allclose
from regions import CircleSkyRegion

from gammapy.data import GTI
from gammapy.datasets import MapDataset, MapDatasetMetaData, MapDatasetOnOff
from gammapy.irf import EDispKernelMap, PSFMap
from gammapy.maps import HpxGeom, Map, MapAxis, RegionGeom, WcsGeom
from gammapy.utils.metadata import (
    CreatorMetaData,
    ObsInfoMetaData,
    PointingInfoMetaData,
)
from gammapy.utils.testing import assert_time_allclose

asdf = pytest.importorskip("asdf")
pytest.importorskip("asdf.testing")


def test_mapdataset_roundtrip(tmp_path):
    file_path = tmp_path / "test.asdf"

    energy_axis = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=2, name="energy")
    energy_axis_true = MapAxis.from_energy_bounds(
        "0.1 TeV", "10 TeV", nbin=3, name="energy_true"
    )
    geom = WcsGeom.create(
        skydir=(266.40498829, -28.93617776),
        binsz=0.5,
        width=(2, 2),
        frame="icrs",
        axes=[energy_axis],
    )

    geom_etrue = WcsGeom.create(
        skydir=(266.40498829, -28.93617776),
        binsz=0.5,
        width=(2, 2),
        frame="icrs",
        axes=[energy_axis_true],
    )

    counts = Map.from_geom(geom)
    counts.data = np.arange(counts.data.size, dtype=counts.data.dtype).reshape(
        counts.geom.data_shape
    )

    exposure = Map.from_geom(geom_etrue, unit="m2 s")
    exposure.data = np.ones(exposure.data.shape) * 1e10

    background = Map.from_geom(geom)
    background.data = np.ones(background.data.shape) * 0.5

    mask_safe = Map.from_geom(geom, dtype="bool")
    mask_safe.data[1:] = True
    edisp = EDispKernelMap.from_gauss(
        geom.axes["energy"], geom_etrue.axes["energy_true"], sigma=0.1, bias=0
    )
    psf = PSFMap.from_gauss(
        energy_axis_true=energy_axis_true,
        sigma=[0.1, 0.2, 0.3] * u.deg,
        geom=geom.to_image(),
    )
    gti = GTI.create([0 * u.s], [1 * u.h], reference_time="2010-01-01T00:00:00")

    center = SkyCoord("0.2 deg", "0.1 deg", frame="galactic")
    circle = CircleSkyRegion(center=center, radius=1 * u.deg)
    mask_fit = geom.region_mask([circle])
    meta_table = Table({"OBS_ID": [1]})
    meta = MapDatasetMetaData(
        creation=CreatorMetaData(
            creator="gammapy",
            date=Time("2020-01-01"),
        )
    )

    dataset = MapDataset(
        counts=counts,
        exposure=exposure,
        background=background,
        mask_safe=mask_safe,
        mask_fit=mask_fit,
        psf=psf,
        edisp=edisp,
        gti=gti,
        meta_table=meta_table,
        meta=meta,
        name="test-mapdataset",
    )

    with asdf.AsdfFile() as af:
        af["dataset"] = dataset
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        result = af["dataset"]
        assert_allclose(result.counts.data, dataset.counts.data)
        assert_allclose(result.exposure.data, dataset.exposure.data)
        assert_allclose(result.background.data, dataset.background.data)
        assert_allclose(result.mask_safe.data, dataset.mask_safe.data)
        assert_allclose(result.mask_fit.data, dataset.mask_fit.data)
        assert_allclose(result.edisp.edisp_map.data, dataset.edisp.edisp_map.data)
        assert_allclose(result.psf.psf_map.data, dataset.psf.psf_map.data)
        assert result.name == dataset.name
        assert result.meta.creation.creator == dataset.meta.creation.creator
        assert_time_allclose(result.meta.creation.date, dataset.meta.creation.date)
        assert_allclose(
            result.gti.time_sum.to_value("s"), dataset.gti.time_sum.to_value("s")
        )
        assert list(result.meta_table["OBS_ID"]) == list(dataset.meta_table["OBS_ID"])
        assert result.models is None


def test_mapdataset_roundtrip_region_geom(tmp_path):
    file_path = tmp_path / "test.asdf"
    energy_axis = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=2, name="energy")
    geom = RegionGeom.create("icrs;circle(266.4, -28.9, 0.5)", axes=[energy_axis])

    counts = Map.from_geom(geom)
    counts.data = np.arange(counts.data.size, dtype=counts.data.dtype).reshape(
        counts.geom.data_shape
    )

    dataset = MapDataset(counts=counts, name="region-test")

    with asdf.AsdfFile() as af:
        af["dataset"] = dataset
        af.write_to(file_path)
    with asdf.open(file_path) as af:
        result = af["dataset"]
        assert_allclose(result.counts.data, dataset.counts.data)


def test_mapdatasetonoff_roundtrip(tmp_path):
    file_path = tmp_path / "test.asdf"

    energy_axis = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=2, name="energy")
    energy_axis_true = MapAxis.from_energy_bounds(
        "0.1 TeV", "10 TeV", nbin=3, name="energy_true"
    )
    geom = WcsGeom.create(
        skydir=(266.40498829, -28.93617776),
        binsz=0.5,
        width=(2, 2),
        frame="galactic",
        axes=[energy_axis],
    )

    geom_etrue = WcsGeom.create(
        skydir=(266.40498829, -28.93617776),
        binsz=0.5,
        width=(2, 2),
        frame="galactic",
        axes=[energy_axis_true],
    )

    counts = Map.from_geom(geom)
    counts.data = np.arange(counts.data.size, dtype=counts.data.dtype).reshape(
        counts.geom.data_shape
    )

    counts_off = Map.from_geom(geom)
    counts_off.data = np.ones(counts_off.data.shape) * 10.0

    acceptance = Map.from_geom(geom)
    acceptance.data = np.arange(
        acceptance.data.size, dtype=acceptance.data.dtype
    ).reshape(acceptance.geom.data_shape)

    acceptance_off = Map.from_geom(geom)
    acceptance_off.data = np.ones(acceptance_off.data.shape) * 5.0

    exposure = Map.from_geom(geom_etrue, unit="m2 s")
    exposure.data = np.ones(exposure.data.shape) * 1e10

    mask_safe = Map.from_geom(geom, dtype="bool")
    mask_safe.data[1:] = True

    edisp = EDispKernelMap.from_gauss(
        geom.axes["energy"], geom_etrue.axes["energy_true"], sigma=0.1, bias=0
    )
    psf = PSFMap.from_gauss(
        energy_axis_true=energy_axis_true,
        sigma=[0.1, 0.2, 0.3] * u.deg,
        geom=geom.to_image(),
    )
    gti = GTI.create([0 * u.s], [1 * u.h], reference_time="2010-01-01T00:00:00")

    center = SkyCoord("0.2 deg", "0.1 deg", frame="galactic")
    circle = CircleSkyRegion(center=center, radius=1 * u.deg)
    mask_fit = geom.region_mask([circle])
    meta_table = Table(
        {
            "OBS_ID": [10, 11, 12],
            "LIVETIME": [1800.0, 1750.5, 1820.3],
        }
    )

    meta = MapDatasetMetaData(
        obs_info=ObsInfoMetaData(**{"instrument": "H.E.S.S.", "obs_id": 112}),
        pointing=PointingInfoMetaData(
            radec_mean=SkyCoord(83.6287, 22.5147, unit="deg", frame="icrs")
        ),
        optional=dict(test=0.5, other=True),
    )

    dataset = MapDatasetOnOff(
        counts=counts,
        counts_off=counts_off,
        acceptance=acceptance,
        acceptance_off=acceptance_off,
        exposure=exposure,
        mask_safe=mask_safe,
        mask_fit=mask_fit,
        psf=psf,
        edisp=edisp,
        gti=gti,
        meta_table=meta_table,
        meta=meta,
        name="test-mapdatasetonoff",
    )

    with asdf.AsdfFile() as af:
        af["dataset"] = dataset
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        result = af["dataset"]
        assert_allclose(result.counts.data, dataset.counts.data)
        assert_allclose(result.counts_off.data, dataset.counts_off.data)
        assert_allclose(result.acceptance.data, dataset.acceptance.data)
        assert_allclose(result.acceptance_off.data, dataset.acceptance_off.data)
        assert_allclose(result.exposure.data, dataset.exposure.data)
        assert_allclose(result.mask_safe.data, dataset.mask_safe.data)
        assert_allclose(result.mask_fit.data, dataset.mask_fit.data)
        assert_allclose(result.edisp.edisp_map.data, dataset.edisp.edisp_map.data)
        assert_allclose(result.psf.psf_map.data, dataset.psf.psf_map.data)
        assert result.name == dataset.name
        assert dataset.meta.obs_info == result.meta.obs_info
        assert dataset.meta.pointing == result.meta.pointing
        assert dataset.meta.optional == result.meta.optional
        assert_allclose(
            result.gti.time_sum.to_value("s"), dataset.gti.time_sum.to_value("s")
        )
        assert list(result.meta_table["OBS_ID"]) == list(dataset.meta_table["OBS_ID"])
        assert_allclose(result.meta_table["LIVETIME"], dataset.meta_table["LIVETIME"])
        assert result.models is None


def test_mapdatasetonoff_roundtrip_hpx_geom(tmp_path):
    file_path = tmp_path / "test.asdf"
    energy_axis = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=2, name="energy")
    geom = HpxGeom.create(nside=8, frame="galactic", axes=[energy_axis])

    counts = Map.from_geom(geom)
    counts.data = np.arange(counts.data.size, dtype=counts.data.dtype).reshape(
        counts.geom.data_shape
    )

    dataset = MapDatasetOnOff(counts=counts, name="hpx-test")

    with asdf.AsdfFile() as af:
        af["dataset"] = dataset
        af.write_to(file_path)
    with asdf.open(file_path) as af:
        result = af["dataset"]
        assert_allclose(result.counts.data, dataset.counts.data)

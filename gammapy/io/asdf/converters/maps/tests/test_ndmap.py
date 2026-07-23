# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
import astropy.units as u

from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from astropy.time import Time
from gammapy.maps import HpxGeom, HpxNDMap, WcsGeom, WcsNDMap, MapAxis

asdf = pytest.importorskip("asdf")
pytest.importorskip("asdf.testing")

axes1 = [MapAxis(np.logspace(0.0, 3.0, 3), interp="log", name="energy")]
axes2 = [
    MapAxis(np.logspace(0.0, 3.0, 3), interp="log", name="energy"),
    MapAxis(np.logspace(1.0, 3.0, 4), interp="lin"),
]
skydir = SkyCoord(110.0, 75.0, unit="deg", frame="icrs")

tested_allsky_wcs_ndmap = [
    (None, 10.0, "galactic", "AIT", skydir, None, "", None),
    (None, 10.0, "icrs", "AIT", skydir, axes1, "", {"telescope": "CTA"}),
    (None, [10.0, 20.0], "galactic", "AIT", skydir, axes1, "cm2 s", None),
    (None, 10.0, "icrs", "AIT", skydir, axes2, "m2", {"telescope": "HESS"}),
    (
        None,
        [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]],
        "galactic",
        "AIT",
        skydir,
        axes2,
        "s",
        None,
    ),
]
tested_partialsky_wcs_ndmap = [
    (
        10,
        1.0,
        "icrs",
        "AIT",
        skydir,
        None,
        "cm2",
        {
            "livetime": 100.0 * u.s,
            "t_start": Time("2020-01-01"),
            "t_stop": Time("2020-01-02"),
        },
    ),
    (
        10,
        1.0,
        "galactic",
        "AIT",
        skydir,
        axes1,
        "",
        {
            "observation_time": Time("2020-01-01"),
            "exposure": 50.0 * u.s,
            "target": skydir,
        },
    ),
    (10, [1.0, 2.0], "icrs", "AIT", skydir, axes1, 1 / (u.cm**2 / u.s), None),
    (10, 1.0, "galactic", "AIT", skydir, axes2, "cm2 s-1", {"creator": "gammapy"}),
    (
        10,
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        "icrs",
        "AIT",
        skydir,
        axes2,
        "m2 s",
        None,
    ),
]

tested_wcs_ndmap = tested_allsky_wcs_ndmap + tested_partialsky_wcs_ndmap


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes", "unit", "meta"),
    tested_wcs_ndmap,
)
def test_wcsndmap_roundtrip(
    npix, binsz, frame, proj, skydir, axes, unit, meta, tmp_path
):
    file_path = tmp_path / "test.asdf"
    geom = WcsGeom.create(
        npix=npix, binsz=binsz, frame=frame, proj=proj, skydir=skydir, axes=axes
    )
    m = WcsNDMap(geom, unit=unit, meta=meta)
    m.data = np.arange(m.data.size, dtype=m.data.dtype).reshape(m.geom.data_shape)
    with asdf.AsdfFile() as af:
        af["map"] = m
        af.write_to(file_path)
    with asdf.open(file_path) as af:
        result = af["map"]
        assert_allclose(result.data, m.data)
        assert result.unit == m.unit
        assert result.meta == m.meta


def test_wcsndmap_roundtrip_compressed(tmp_path):
    file_path = tmp_path / "test.asdf"
    geom = WcsGeom.create(
        npix=10, binsz=1.0, proj="AIT", skydir=skydir, frame="galactic", axes=axes1
    )
    m = WcsNDMap(
        geom,
        unit="m2",
        meta={
            "livetime": 100.0 * u.s,
            "t_start": Time("2020-01-01"),
            "t_stop": Time("2020-01-02"),
        },
    )
    m.data = np.arange(m.data.size, dtype=m.data.dtype).reshape(m.geom.data_shape)
    with asdf.AsdfFile() as af:
        af["map"] = m
        af.write_to(file_path, all_array_compression="zlib")
    with asdf.open(file_path) as af:
        result = af["map"]
        assert_allclose(result.data, m.data)
        assert result.unit == m.unit
        assert result.meta == m.meta


tested_wcs_ndmap_dtypes = [
    (np.float32, "m2", None),
    (np.float64, "", {"test": "dtype"}),
    (bool, "", None),
]


@pytest.mark.parametrize(
    ("dtype", "unit", "meta"),
    tested_wcs_ndmap_dtypes,
)
def test_wcsndmap_roundtrip_dtype(dtype, unit, meta, tmp_path):
    file_path = tmp_path / "test.asdf"
    geom = WcsGeom.create(
        npix=10, binsz=1.0, proj="AIT", skydir=skydir, frame="galactic", axes=axes1
    )
    m = WcsNDMap(geom, unit=unit, meta=meta)
    if dtype is bool:
        m.data = np.arange(m.data.size).reshape(m.geom.data_shape) % 2 == 0
    else:
        m.data = np.arange(m.data.size, dtype=dtype).reshape(m.geom.data_shape)
    with asdf.AsdfFile() as af:
        af["map"] = m
        af.write_to(file_path)
    with asdf.open(file_path) as af:
        result = af["map"]
        assert_allclose(result.data, m.data)
        assert result.unit == m.unit
        assert result.meta == m.meta
        assert result.data.dtype == m.data.dtype


tested_allsky_hpx_ndmap = [
    (8, False, "galactic", None, None, "", None),
    (
        8,
        False,
        "galactic",
        None,
        [MapAxis(np.logspace(0.0, 3.0, 4))],
        "",
        {"telescope": "CTA"},
    ),
    (
        [2, 4, 8],
        False,
        "galactic",
        None,
        [MapAxis(np.logspace(0.0, 3.0, 4))],
        "m2 s",
        None,
    ),
    (
        8,
        False,
        "galactic",
        None,
        [
            MapAxis(np.logspace(0.0, 3.0, 3), name="axis0"),
            MapAxis(np.logspace(0.0, 2.0, 4), name="axis1"),
        ],
        1 / (u.cm**2 / u.s),
        None,
    ),
    (8, True, "galactic", None, None, "cm2 s-1", {"author": "test"}),
    (8, False, "icrs", None, None, "m2", {"telescope": "HESS"}),
]
tested_partialsky_hpx_ndmap = [
    (8, True, "galactic", "DISK(110.,75.,10.)", None, "", None),
    (
        8,
        False,
        "galactic",
        "DISK(110.,75.,10.)",
        [MapAxis(np.logspace(0.0, 3.0, 4))],
        "",
        {"key": "value"},
    ),
    (
        [8, 16, 32],
        False,
        "galactic",
        "DISK(110.,75.,10.)",
        [MapAxis(np.logspace(0.0, 3.0, 4))],
        "cm2 s",
        None,
    ),
    (
        [[8, 16, 32], [8, 8, 16]],
        False,
        "galactic",
        "DISK(110.,75.,10.)",
        [
            MapAxis(np.logspace(0.0, 3.0, 3), name="axis0"),
            MapAxis(np.logspace(0.0, 2.0, 4), name="axis1"),
        ],
        "s",
        None,
    ),
    (8, True, "icrs", "DISK_INC(110.,75.,10.,4)", None, "", {"creator": "gammapy"}),
    (8, True, "icrs", "HPX_PIXEL(NESTED,2,3)", None, "cm2 s-1", None),
    (8, False, "icrs", "HPX_PIXEL(RING,2,3)", None, "", None),
]
tested_hpx_ndmap = tested_allsky_hpx_ndmap + tested_partialsky_hpx_ndmap


@pytest.mark.parametrize(
    ("nside", "nested", "frame", "region", "axes", "unit", "meta"), tested_hpx_ndmap
)
def test_hpxndmap_roundtrip(nside, nested, frame, region, axes, unit, meta, tmp_path):
    file_path = tmp_path / "test.asdf"
    geom = HpxGeom(nside=nside, nest=nested, frame=frame, region=region, axes=axes)
    m = HpxNDMap(geom, unit=unit, meta=meta)
    m.data = np.arange(m.data.size, dtype=m.data.dtype).reshape(m.geom.data_shape)
    with asdf.AsdfFile() as af:
        af["map"] = m
        af.write_to(file_path)
    with asdf.open(file_path) as af:
        result = af["map"]
        assert_allclose(result.data, m.data)
        assert result.unit == m.unit
        assert result.meta == m.meta


def test_hpxndmap_roundtrip_compressed(tmp_path):
    file_path = tmp_path / "test.asdf"
    geom = HpxGeom(
        nside=8, nest=False, frame="galactic", region="DISK(110.,75.,10.)", axes=None
    )
    idx = geom.get_idx(flat=True)
    geom = HpxGeom(nside=8, nest=False, frame="galactic", region=idx, axes=None)
    m = HpxNDMap(
        geom,
        unit=1 / (u.cm**2 / u.s),
        meta={
            "observation_date": Time("2020-01-15"),
            "exposure": 7200.0 * u.s,
            "telescope": "Fermi",
        },
    )
    m.data = np.arange(m.data.size, dtype=m.data.dtype).reshape(m.geom.data_shape)
    with asdf.AsdfFile() as af:
        af["map"] = m
        af.write_to(file_path, all_array_compression="zlib")
    with asdf.open(file_path) as af:
        result = af["map"]
        assert_allclose(result.data, m.data)
        assert result.unit == m.unit
        assert result.meta == m.meta


tested_hpx_ndmap_dtypes = [
    (np.float32, "m2", None),
    (np.float64, "", {"test": "dtype"}),
    (bool, "", None),
]


@pytest.mark.parametrize(
    ("dtype", "unit", "meta"),
    tested_hpx_ndmap_dtypes,
)
def test_hpxndmap_roundtrip_dtype(dtype, unit, meta, tmp_path):
    file_path = tmp_path / "test.asdf"
    geom = HpxGeom(
        nside=8, nest=False, frame="galactic", region="DISK(110.,75.,10.)", axes=axes1
    )
    m = HpxNDMap(geom, unit=unit, meta=meta)
    if dtype is bool:
        m.data = np.arange(m.data.size).reshape(m.geom.data_shape) % 2 == 0
    else:
        m.data = np.arange(m.data.size, dtype=dtype).reshape(m.geom.data_shape)
    with asdf.AsdfFile() as af:
        af["map"] = m
        af.write_to(file_path)
    with asdf.open(file_path) as af:
        result = af["map"]
        assert_allclose(result.data, m.data)
        assert result.unit == m.unit
        assert result.meta == m.meta
        assert result.data.dtype == m.data.dtype

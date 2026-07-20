# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
import astropy.units as u

from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from astropy.time import Time
from gammapy.maps import WcsGeom, WcsNDMap, MapAxis

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
    m.data = np.arange(m.data.size).reshape(m.geom.data_shape)
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
    m.data = np.arange(m.data.size).reshape(m.geom.data_shape)
    with asdf.AsdfFile() as af:
        af["map"] = m
        af.write_to(file_path, all_array_compression="zlib")
    with asdf.open(file_path) as af:
        result = af["map"]
        assert_allclose(result.data, m.data)
        assert result.unit == m.unit
        assert result.meta == m.meta

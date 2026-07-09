# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose
from gammapy.maps import MapAxis, WcsGeom, HpxGeom

asdf = pytest.importorskip("asdf")
pytest.importorskip("asdf.testing")
from asdf.testing.helpers import yaml_to_asdf  # noqa: E402

axes1 = [MapAxis(np.logspace(0.0, 3.0, 3), interp="log", name="energy")]
axes2 = [
    MapAxis(np.logspace(0.0, 3.0, 3), interp="log", name="energy"),
    MapAxis(np.logspace(1.0, 3.0, 4), interp="lin"),
]
skydir = SkyCoord(110.0, 75.0, unit="deg", frame="icrs")

tested_wcs_geom = [
    (None, 10.0, "galactic", "AIT", skydir, None),
    (None, 10.0, "icrs", "AIT", skydir, axes1),
    (None, [10.0, 20.0], "galactic", "AIT", skydir, axes1),
    (None, 10.0, "icrs", "AIT", skydir, axes2),
    (None, [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]], "galactic", "AIT", skydir, axes2),
    (10, 0.1, "icrs", "AIT", skydir, None),
    (10, 0.1, "galactic", "AIT", skydir, axes1),
    (10, [0.1, 0.2], "icrs", "AIT", skydir, axes1),
]


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), tested_wcs_geom
)
def test_wcsgeom_roundtrip(npix, binsz, frame, proj, skydir, axes, tmp_path):
    file_path = tmp_path / "test.asdf"
    geom = WcsGeom.create(
        npix=npix,
        binsz=binsz,
        frame=frame,
        proj=proj,
        skydir=skydir,
        axes=axes,
    )
    with asdf.AsdfFile() as af:
        af["geom"] = geom
        af.write_to(file_path)
    with asdf.open(file_path) as af:
        result = af["geom"]
        assert result.is_allclose(geom)
        if not geom.is_regular:
            assert_allclose(result._cdelt[0], geom._cdelt[0])
            assert_allclose(result._cdelt[1], geom._cdelt[1])
            assert_allclose(result._crpix[0], geom._crpix[0])
            assert_allclose(result._crpix[1], geom._crpix[1])


def test_wcs_geom_roundtrip_offcenter(tmp_path):
    from astropy.wcs import WCS

    file_path = tmp_path / "test.asdf"
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    w.wcs.crval = [83.63, 22.01]
    w.wcs.cdelt = [-0.02, 0.02]
    w.wcs.crpix = [30.0, 45.0]
    w.array_shape = (100, 80)
    geom = WcsGeom(wcs=w)
    with asdf.AsdfFile() as af:
        af["geom"] = geom
        af.write_to(file_path)
    with asdf.open(file_path) as af:
        result = af["geom"]
        assert result.is_allclose(geom)
        assert_allclose(result.wcs.wcs.crpix, [30.0, 45.0])


def test_wcs_geom_invalid():
    example = """!<asdf://gammapy.org/gammapy/tags/maps/wcsgeom-1.0.0>
            npix:
              - !core/ndarray-1.1.0
                data: [50]
              - !core/ndarray-1.1.0
                data: [100]
             """

    buff = yaml_to_asdf(f"example: {example.strip()}")
    with pytest.raises(asdf.exceptions.ValidationError):
        asdf.open(buff)


tested_hpx_geom = [
    # All-sky
    (8, False, "galactic", None, None),
    (8, False, "galactic", None, [MapAxis(np.logspace(0.0, 3.0, 4))]),
    ([2, 4, 8], False, "galactic", None, [MapAxis(np.logspace(0.0, 3.0, 4))]),
    (
        8,
        False,
        "galactic",
        None,
        [
            MapAxis(np.logspace(0.0, 3.0, 3), name="axis0"),
            MapAxis(np.logspace(0.0, 2.0, 4), name="axis1"),
        ],
    ),
    (8, True, "galactic", None, None),
    (8, False, "icrs", None, None),
    # Partial-sky
    (8, False, "galactic", "DISK(110.,75.,10.)", None),
    (8, False, "galactic", "DISK(110.,75.,10.)", [MapAxis(np.logspace(0.0, 3.0, 4))]),
    (
        [8, 16, 32],
        False,
        "galactic",
        "DISK(110.,75.,10.)",
        [MapAxis(np.logspace(0.0, 3.0, 4))],
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
    ),
    (8, True, "galactic", "DISK(110.,75.,10.)", None),
    (8, False, "icrs", "DISK_INC(110.,75.,10.,4)", None),
    (8, True, "icrs", "HPX_PIXEL(NESTED,2,3)", None),
    (8, False, "icrs", "HPX_PIXEL(RING,2,3)", None),
]


@pytest.mark.parametrize(
    ("nside", "nested", "frame", "region", "axes"), tested_hpx_geom
)
def test_hpxgeom_roundtrip(nside, nested, frame, region, axes, tmp_path):
    file_path = tmp_path / "test.asdf"
    geom = HpxGeom(nside, nested, frame, region=region, axes=axes)
    with asdf.AsdfFile() as af:
        af["geom"] = geom
        af.write_to(file_path)
    with asdf.open(file_path) as af:
        result = af["geom"]
        if geom.nside.size > 1:
            assert_allclose(result.nside, geom.nside)
            assert result.nest == geom.nest
            assert result.frame == geom.frame
            assert result.region == geom.region
            assert result.axes == geom.axes
        else:
            assert result == geom
            assert result.region == geom.region


def test_hpxgeom_roundtrip_tuple_region(tmp_path):
    file_path = tmp_path / "test.asdf"
    geom = HpxGeom(
        nside=8, nest=False, frame="galactic", region="DISK(110.,75.,10.)", axes=None
    )
    idx = geom.get_idx(flat=True)
    geom = HpxGeom(nside=8, nest=False, frame="galactic", region=idx, axes=None)
    with asdf.AsdfFile() as af:
        af["geom"] = geom
        af.write_to(file_path)
    with asdf.open(file_path) as af:
        result = af["geom"]
        assert result == geom
        assert result.region == geom.region
        assert_allclose(result._ipix, geom._ipix)


tested_read_hpxgeom_examples = [
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/hpxgeom-1.0.0>
          axes: !<asdf://gammapy.org/gammapy/tags/maps/mapaxes-1.0.0>
            axes: []
          frame: galactic
          nest: false
          nside: !core/ndarray-1.1.0
            data: [8]
          region: DISK(110.,75.,10.)
          """,
        "truth": HpxGeom(
            nside=8,
            nest=False,
            frame="galactic",
            region="DISK(110.,75.,10.)",
            axes=None,
        ),
    },
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/hpxgeom-1.0.0>
          axes: !<asdf://gammapy.org/gammapy/tags/maps/mapaxes-1.0.0>
            axes: []
          nest: false
          frame: galactic
          nside: !core/ndarray-1.1.0
            data: [8]
          ipix: !core/ndarray-1.1.0
            data: [6, 15, 16, 28, 29]
          region: explicit
          """,
        "truth": HpxGeom(
            nside=8,
            nest=False,
            frame="galactic",
            region=(np.array([6, 15, 16, 28, 29]),),
            axes=None,
        ),
    },
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/hpxgeom-1.0.0>
          axes: !<asdf://gammapy.org/gammapy/tags/maps/mapaxes-1.0.0>
            axes: []
          frame: icrs
          nest: true
          nside: !core/ndarray-1.1.0
            data: [8]
          """,
        "truth": HpxGeom(nside=8, nest=True, region=None, axes=None),
    },
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/hpxgeom-1.0.0>
          axes: !<asdf://gammapy.org/gammapy/tags/maps/mapaxes-1.0.0>
            axes:
            - !<asdf://gammapy.org/gammapy/tags/maps/mapaxis-1.0.0>
              boundary_type: monotonic
              interp: log
              name: energy
              node_type: edges
              nodes: !core/ndarray-1.1.0 [1.0, 10.0, 100.0]
              unit: TeV
          frame: icrs
          nest : true
          nside: !core/ndarray-1.1.0
            data: [4, 8]
          """,
        "truth": HpxGeom(
            nside=np.array([4, 8]),
            nest=True,
            frame="icrs",
            axes=[
                MapAxis(
                    nodes=[1.0, 10.0, 100.0],
                    unit="TeV",
                    name="energy",
                    interp="log",
                    node_type="edges",
                )
            ],
        ),
    },
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/hpxgeom-1.0.0>
          nside: !core/ndarray-1.1.0
            data: [16]
          nest: invalid
          frame: icrs """,
    },
    {
        "example": """!<asdf://gammapy.org/gammapy/tags/maps/hpxgeom-1.0.0>
          nest: true
          frame: icrs
          """,
    },
]


@pytest.mark.parametrize("example", tested_read_hpxgeom_examples)
def test_hpx_geom_read_examples(example):
    buff = yaml_to_asdf(f"example: {example['example'].strip()}")

    if example.get("truth") is not None:
        truth = example["truth"]
        with asdf.open(buff) as af:
            result = af["example"]
            if truth.nside.size > 1:
                assert_allclose(result.nside, truth.nside)
                assert result.nest == truth.nest
                assert result.frame == truth.frame
                assert result.region == truth.region
                assert result.axes == truth.axes
            else:
                assert result == truth
                assert result.region == truth.region
    else:
        with pytest.raises(asdf.exceptions.ValidationError):
            asdf.open(buff)

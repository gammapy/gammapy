# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from ..geom import MapAxis
from ..hpx import HpxGeom
from ..hpxsparse import HpxSparseMap

pytest.importorskip("healpy")

hpx_test_geoms = [
    (8, False, "GAL", None, None),
    (8, False, "GAL", None, [MapAxis(np.logspace(0.0, 3.0, 4))]),
    (8, False, "GAL", "DISK(110.,75.,10.)", [MapAxis(np.logspace(0.0, 3.0, 4))]),
    (
        [8, 16, 32],
        False,
        "GAL",
        "DISK(110.,75.,10.)",
        [MapAxis(np.logspace(0.0, 3.0, 4))],
    ),
    (
        8,
        False,
        "GAL",
        "DISK(110.,75.,10.)",
        [
            MapAxis(np.logspace(0.0, 3.0, 4), name="axis0"),
            MapAxis(np.logspace(0.0, 2.0, 3), name="axis1"),
        ],
    ),
]


@pytest.mark.parametrize(
    ("nside", "nested", "coordsys", "region", "axes"), hpx_test_geoms
)
def test_hpxsparse_init(nside, nested, coordsys, region, axes):
    geom = HpxGeom(nside, nested, coordsys, region=region, axes=axes)
    HpxSparseMap(geom)
    # TODO: Test initialization w/ data array

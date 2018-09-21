# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from ...utils.testing import requires_data
from ...maps import WcsGeom, HpxGeom, MapAxis
from ...irf import Background3D
from ..background import make_map_background_irf

pytest.importorskip("scipy")
pytest.importorskip("healpy")


@pytest.fixture(scope="session")
def bkg_3d():
    filename = "$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    return Background3D.read(filename, hdu="BACKGROUND")


def geom(map_type, ebounds):
    axis = MapAxis.from_edges(ebounds, name="energy", unit="TeV", interp="log")
    if map_type == "wcs":
        return WcsGeom.create(npix=(4, 3), binsz=2, axes=[axis])
    elif map_type == "hpx":
        return HpxGeom(256, axes=[axis])
    else:
        raise ValueError()


@requires_data("gammapy-extra")
@pytest.mark.parametrize(
    "pars",
    [
        {
            "geom": geom(map_type="wcs", ebounds=[0.1, 1, 10]),
            "shape": (2, 3, 4),
            "sum": 4760.562826,
        },
        {
            "geom": geom(map_type="wcs", ebounds=[0.1, 10]),
            "shape": (1, 3, 4),
            "sum": 49620.735907,
        },
        # TODO: make this work for HPX
        # 'HpxGeom' object has no attribute 'separation'
        # {
        #     'geom': geom(map_type='hpx', ebounds=[0.1, 1, 10]),
        #     'shape': '???',
        #     'sum': '???',
        # },
    ],
)
def test_make_map_background_irf(bkg_3d, pars):
    m = make_map_background_irf(
        pointing=SkyCoord(2, 1, unit="deg"),
        livetime="42 s",
        bkg=bkg_3d,
        geom=pars["geom"],
    )

    assert m.data.shape == pars["shape"]
    assert m.unit == ""
    assert_allclose(m.data.sum(), pars["sum"], rtol=1e-5)

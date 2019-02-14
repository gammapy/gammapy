# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from ...utils.testing import requires_data
from ...maps import WcsGeom, HpxGeom, MapAxis
from ...irf import Background3D
from ..background import make_map_background_irf
from ...data.pointing import FixedPointingInfo

pytest.importorskip("healpy")

@pytest.fixture(scope="session")
def fixed_pointing_info():
    filename = (
        "$GAMMAPY_DATA/cta-1dc/data/baseline/gps/gps_baseline_110380.fits"
    )
    return FixedPointingInfo.read(filename)


@pytest.fixture(scope="session")
def bkg_3d():
    filename = (
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    return Background3D.read(filename, hdu="BACKGROUND")


@pytest.fixture(scope="session")
def bkg_3d_asymmetric():
    """Example with simple values to test evaluate"""
    energy = [0.1, 10, 1000] * u.TeV
    fov_lon = [0, 1, 2, 3] * u.deg
    fov_lat = [0, 1, 2, 3] * u.deg

    data = np.ones((2, 3, 3)) * u.Unit("s-1 MeV-1 sr-1")
    data *= np.arange(1, 4).reshape(1, 3, 1)
    return Background3D(
        energy_lo=energy[:-1],
        energy_hi=energy[1:],
        fov_lon_lo=fov_lon[:-1],
        fov_lon_hi=fov_lon[1:],
        fov_lat_lo=fov_lat[:-1],
        fov_lat_hi=fov_lat[1:],
        data=data,
    )


def geom(map_type, ebounds, skydir):
    axis = MapAxis.from_edges(ebounds, name="energy", unit="TeV", interp="log")
    if map_type == "wcs":
        return WcsGeom.create(npix=(4, 3), binsz=2, axes=[axis], skydir=skydir)
    elif map_type == "hpx":
        return HpxGeom(256, axes=[axis])
    else:
        raise ValueError()


@requires_data("gammapy-data")
@pytest.mark.parametrize(
    "pars",
    [
        {
            "map_type": "wcs",
            "ebounds": [0.1, 1, 10],
            "shape": (2, 3, 4),
            "sum": 928.955085,
        },
        {
            "map_type": "wcs",
            "ebounds": [0.1, 10],
            "shape": (1, 3, 4),
            "sum": 1006.720592,
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

def test_make_map_background_irf(bkg_3d, pars, fixed_pointing_info):
    m = make_map_background_irf(
        pointing=fixed_pointing_info,
        ontime="42 s",
        bkg=bkg_3d,
        geom=geom(
            map_type=pars['map_type'],
            ebounds=pars['ebounds'],
            skydir=fixed_pointing_info.radec
        ),
    )

    assert m.data.shape == pars["shape"]
    assert m.unit == ""
    assert_allclose(m.data.sum(), pars["sum"], rtol=1e-5)

def test_make_map_background_irf_asym(bkg_3d_asymmetric, fixed_pointing_info):
    m = make_map_background_irf(
        pointing=fixed_pointing_info,
        ontime="42 s",
        bkg=bkg_3d_asymmetric,
        geom=geom(
            map_type="wcs",
            ebounds=[0.1, 1, 10],
            skydir=fixed_pointing_info.radec
        ),
    )
    assert_allclose(m.data[0, 0, 0], 34420.569872, rtol=1e-5)
    assert_allclose(m.data[0, 1, 2], 47216.385951, rtol=1e-5)

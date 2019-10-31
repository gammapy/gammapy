# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from gammapy.cube.exposure import _map_spectrum_weight, make_map_exposure_true_energy
from gammapy.irf import EffectiveAreaTable2D
from gammapy.maps import HpxGeom, MapAxis, WcsGeom, WcsNDMap
from gammapy.modeling.models import ConstantSpectralModel
from gammapy.utils.testing import requires_data

pytest.importorskip("healpy")


@pytest.fixture(scope="session")
def aeff():
    filename = (
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    return EffectiveAreaTable2D.read(filename, hdu="EFFECTIVE AREA")


def geom(map_type, ebounds):
    axis = MapAxis.from_edges(ebounds, name="energy", unit="TeV", interp="log")
    if map_type == "wcs":
        return WcsGeom.create(npix=(4, 3), binsz=2, axes=[axis])
    elif map_type == "hpx":
        return HpxGeom(256, axes=[axis])
    else:
        raise ValueError()


@requires_data()
@pytest.mark.parametrize(
    "pars",
    [
        {
            "geom": geom(map_type="wcs", ebounds=[0.1, 1, 10]),
            "shape": (2, 3, 4),
            "sum": 8.103974e08,
        },
        {
            "geom": geom(map_type="wcs", ebounds=[0.1, 10]),
            "shape": (1, 3, 4),
            "sum": 2.387916e08,
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
def test_make_map_exposure_true_energy(aeff, pars):
    m = make_map_exposure_true_energy(
        pointing=SkyCoord(2, 1, unit="deg"),
        livetime="42 s",
        aeff=aeff,
        geom=pars["geom"],
    )

    assert m.data.shape == pars["shape"]
    assert m.unit == "m2 s"
    assert_allclose(m.data.sum(), pars["sum"], rtol=1e-5)


def test_map_spectrum_weight():
    axis = MapAxis.from_edges([0.1, 10, 1000], unit="TeV", name="energy")
    expo_map = WcsNDMap.create(npix=10, binsz=1, axes=[axis], unit="m2 s")
    expo_map.data += 1
    spectrum = ConstantSpectralModel(const="42 cm-2 s-1 TeV-1")

    weighted_expo = _map_spectrum_weight(expo_map, spectrum)

    assert weighted_expo.data.shape == (2, 10, 10)
    assert weighted_expo.unit == "m2 s"
    assert_allclose(weighted_expo.data.sum(), 100)

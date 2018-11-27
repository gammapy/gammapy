# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from ...utils.testing import requires_data
from ...data import DataStore
from ...maps import WcsGeom, MapAxis, Map
from ..make import MapMaker


@pytest.fixture(scope="session")
def observations():
    data_store = DataStore.from_dir("$GAMMAPY_EXTRA/datasets/cta-1dc/index/gps/")
    obs_id = [110380, 111140]
    return data_store.get_observations(obs_id)


def geom(ebounds):
    skydir = SkyCoord(0, -1, unit="deg", frame="galactic")
    energy_axis = MapAxis.from_edges(ebounds, name="energy", unit="TeV", interp="log")
    return WcsGeom.create(
        binsz=0.5 * u.deg,
        skydir=skydir,
        width=(10, 5),
        coordsys="GAL",
        axes=[energy_axis],
    )


@requires_data("gammapy-extra")
@pytest.mark.parametrize(
    "pars",
    [
        {
            # Default, normal test case
            "geom": geom(ebounds=[0.1, 1, 10]),
            "geom_true": None,
            "counts": 34366,
            "exposure": 3.99815e11,
            "exposure_image": 7.921993e10,
            "background": 27984.78,
        },
        {
            # Test single energy bin
            "geom": geom(ebounds=[0.1, 10]),
            "geom_true": None,
            "counts": 34366,
            "exposure": 1.16866e11,
            "exposure_image": 1.16866e11,
            "background": 30418.951,
        },
        {
            # Test single energy bin with exclusion mask
            "geom": geom(ebounds=[0.1, 10]),
            "geom_true": None,
            "exclusion_mask": Map.from_geom(geom(ebounds=[0.1, 10])),
            "counts": 34366,
            "exposure": 1.16866e11,
            "exposure_image": 1.16866e11,
            "background": 30418.951,
        },
        {
            # Test for different e_true and e_reco bins
            "geom": geom(ebounds=[0.1, 1, 10]),
            "geom_true": geom(ebounds=[0.1, 0.5, 2.5, 10.0]),
            "counts": 34366,
            "exposure": 5.971096e11,
            "exposure_image": 6.492968e10,
            "background": 27984.78,
        },
    ],
)
@pytest.mark.parametrize("keepdims", [True, False])
def test_map_maker(pars, observations, keepdims):
    maker = MapMaker(geom=pars["geom"], geom_true=pars["geom_true"], offset_max="2 deg")

    maps = maker.run(observations)

    counts = maps["counts"]
    assert counts.unit == ""
    assert_allclose(counts.data.sum(), pars["counts"], rtol=1e-5)

    exposure = maps["exposure"]
    assert exposure.unit == "m2 s"
    assert_allclose(exposure.data.sum(), pars["exposure"], rtol=1e-5)

    background = maps["background"]
    assert background.unit == ""
    assert_allclose(background.data.sum(), pars["background"], rtol=1e-5)

    images = maker.make_images(keepdims=keepdims)

    counts = images["counts"]
    assert counts.unit == ""
    assert_allclose(counts.data.sum(), pars["counts"], rtol=1e-5)

    exposure = images["exposure"]
    assert exposure.unit == "m2 s"
    assert_allclose(exposure.data.sum(), pars["exposure_image"], rtol=1e-5)

    background = images["background"]
    assert background.unit == ""
    assert_allclose(background.data.sum(), pars["background"], rtol=1e-5)

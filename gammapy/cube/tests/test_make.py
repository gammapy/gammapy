# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from gammapy.cube import MapMaker, MapMakerObs, MapMakerRing, RingBackgroundEstimator
from gammapy.data import DataStore
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def observations():
    data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
    obs_id = [110380, 111140]
    return data_store.get_observations(obs_id)


def geom(ebounds, binsz=0.5):
    skydir = SkyCoord(0, -1, unit="deg", frame="galactic")
    energy_axis = MapAxis.from_edges(ebounds, name="energy", unit="TeV", interp="log")
    return WcsGeom.create(
        binsz=binsz, skydir=skydir, width=(10, 5), coordsys="GAL", axes=[energy_axis]
    )


@requires_data()
@pytest.mark.parametrize(
    "pars",
    [
        {
            # Default, normal test case
            "geom": geom(ebounds=[0.1, 1, 10]),
            "geom_true": None,
            "counts": 34366,
            "exposure": 9.995376e08,
            "exposure_image": 7.921993e10,
            "background": 27989.05,
        },
        {
            # Test single energy bin
            "geom": geom(ebounds=[0.1, 10]),
            "geom_true": None,
            "counts": 34366,
            "exposure": 5.843302e08,
            "exposure_image": 1.16866e11,
            "background": 30424.451,
        },
        {
            # Test single energy bin with exclusion mask
            "geom": geom(ebounds=[0.1, 10]),
            "geom_true": None,
            "exclusion_mask": Map.from_geom(geom(ebounds=[0.1, 10])),
            "counts": 34366,
            "exposure": 5.843302e08,
            "exposure_image": 1.16866e11,
            "background": 30424.451,
        },
        {
            # Test for different e_true and e_reco bins
            "geom": geom(ebounds=[0.1, 1, 10]),
            "geom_true": geom(ebounds=[0.1, 0.5, 2.5, 10.0]),
            "counts": 34366,
            "exposure": 9.951827e08,
            "exposure_image": 6.492968e10,
            "background": 28760.283,
            "background_oversampling": 2,
        },
        {
            # Test for different e_true and e_reco bins
            "geom": geom(ebounds=[0.1, 1, 10]),
            "geom_true": geom(ebounds=[0.1, 0.5, 2.5, 10.0], binsz=1),
            "counts": 34366,
            "exposure": 9.951827e08,
            "exposure_image": 6.492968e10,
            "background": 28760.283,
            "background_oversampling": 2,
        },
    ],
)
@pytest.mark.parametrize("keepdims", [True, False])
def test_map_maker(pars, observations, keepdims):
    maker = MapMaker(
        geom=pars["geom"],
        geom_true=pars["geom_true"],
        offset_max="2 deg",
        background_oversampling=pars.get("background_oversampling"),
    )

    maps = maker.run(observations)

    counts = maps["counts"]
    assert counts.unit == ""
    assert_allclose(counts.data.sum(), pars["counts"], rtol=1e-5)

    exposure = maps["exposure"]
    assert exposure.unit == "m2 s"
    assert_allclose(exposure.data.mean(), pars["exposure"], rtol=3e-3)

    background = maps["background"]
    assert background.unit == ""
    assert_allclose(background.data.sum(), pars["background"], rtol=1e-5)

    images = maker.run_images(keepdims=keepdims)

    counts = images["counts"]
    assert counts.unit == ""
    assert_allclose(counts.data.sum(), pars["counts"], rtol=1e-5)

    exposure = images["exposure"]
    assert exposure.unit == "m2 s"
    assert_allclose(exposure.data.sum(), pars["exposure_image"], rtol=3e-3)

    background = images["background"]
    assert background.unit == ""
    assert_allclose(background.data.sum(), pars["background"], rtol=1e-5)


@requires_data()
def test_map_maker_ring(observations):
    ring_bkg = RingBackgroundEstimator(r_in="0.5 deg", width="0.4 deg")
    geomd = geom(ebounds=[0.1, 1, 10])

    mask = Map.from_geom(geomd)

    regions = CircleSkyRegion(
        SkyCoord(0, 0, unit="deg", frame="galactic"), radius=0.5 * u.deg
    )
    mask.data = mask.geom.region_mask([regions], inside=False)

    maker = MapMakerRing(geomd, 2.0 * u.deg, mask, ring_bkg)

    maps = maker.run(observations)
    assert_allclose(np.nansum(maps["on"].data), 34366, rtol=1e-2)
    assert_allclose(np.nansum(maps["exposure_off"].data), 12362.756, rtol=1e-2)
    assert not maps["on"].geom.is_image

    images = maker.run_images(observations)
    assert_allclose(np.nansum(images["on"].data), 34366, rtol=1e-2)
    assert_allclose(np.nansum(images["exposure_off"].data), 163730.62, rtol=1e-2)
    assert images["on"].geom.is_image


@requires_data()
def test_map_maker_obs(observations):
    # Test for different spatial geoms and etrue, ereco bins

    geom_reco = geom(ebounds=[0.1, 1, 10])
    geom_true = geom(ebounds=[0.1, 0.5, 2.5, 10.0], binsz=1.0)
    geom_exp = geom(ebounds=[0.1, 0.5, 2.5, 10.0])
    maker_obs = MapMakerObs(
        observation=observations[0],
        geom=geom_reco,
        geom_true=geom_true,
        offset_max=2.0 * u.deg,
        cutout=False
    )

    map_dataset = maker_obs.run()
    assert map_dataset.counts.geom == geom_reco
    assert map_dataset.background_model.map.geom == geom_reco
    assert map_dataset.exposure.geom == geom_exp
    assert map_dataset.edisp.edisp_map.data.shape == (3, 48, 5, 10)
    assert map_dataset.edisp.exposure_map.data.shape == (3, 1, 5, 10)
    assert map_dataset.psf.psf_map.data.shape == (3, 66, 5, 10)
    assert map_dataset.psf.exposure_map.data.shape == (3, 1, 5, 10)
    assert_allclose(map_dataset.gti.time_delta, 1800.0 * u.s)
    assert map_dataset.name == "obs_110380"

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from gammapy.cube import MapDataset, MapDatasetMaker, RingBackgroundMaker
from gammapy.cube.fit import MapDatasetOnOff
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
        skydir=skydir, binsz=binsz, width=(10, 5), coordsys="GAL", axes=[energy_axis]
    )


@requires_data()
@pytest.mark.parametrize(
    "pars",
    [
        {
            # Default, same e_true and reco
            "geom": geom(ebounds=[0.1, 1, 10]),
            "e_true": None,
            "counts": 34366,
            "exposure": 9.995376e08,
            "exposure_image": 7.921993e10,
            "background": 27989.05,
            "binsz_irf": 0.5,
            "margin_irf": 0.0,
        },
        {
            # Test single energy bin
            "geom": geom(ebounds=[0.1, 10]),
            "e_true": None,
            "counts": 34366,
            "exposure": 5.843302e08,
            "exposure_image": 1.16866e11,
            "background": 30424.451,
            "binsz_irf": 0.5,
            "margin_irf": 0.0,
        },
        {
            # Test single energy bin with exclusion mask
            "geom": geom(ebounds=[0.1, 10]),
            "e_true": None,
            "exclusion_mask": Map.from_geom(geom(ebounds=[0.1, 10])),
            "counts": 34366,
            "exposure": 5.843302e08,
            "exposure_image": 1.16866e11,
            "background": 30424.451,
            "binsz_irf": 0.5,
            "margin_irf": 0.0,
        },
        {
            # Test for different e_true and e_reco bins
            "geom": geom(ebounds=[0.1, 1, 10]),
            "e_true": MapAxis.from_edges(
                [0.1, 0.5, 2.5, 10.0], name="energy", unit="TeV", interp="log"
            ),
            "counts": 34366,
            "exposure": 9.951827e08,
            "exposure_image": 6.492968e10,
            "background": 28760.283,
            "background_oversampling": 2,
            "binsz_irf": 0.5,
            "margin_irf": 0.0,
        },
        {
            # Test for different e_true and e_reco and spatial bins
            "geom": geom(ebounds=[0.1, 1, 10]),
            "e_true": MapAxis.from_edges(
                [0.1, 0.5, 2.5, 10.0], name="energy", unit="TeV", interp="log"
            ),
            "counts": 34366,
            "exposure": 9.951827e08,
            "exposure_image": 6.492968e10,
            "background": 28760.283,
            "background_oversampling": 2,
            "binsz_irf": 1.0,
            "margin_irf": 0.0,
        },
    ],
)
def test_map_maker(pars, observations):
    stacked = MapDataset.create(
        geom=pars["geom"],
        energy_axis_true=pars["e_true"],
        binsz_irf=pars["binsz_irf"],
        margin_irf=pars["margin_irf"],
    )

    for obs in observations:
        maker = MapDatasetMaker(
            geom=pars["geom"],
            energy_axis_true=pars["e_true"],
            offset_max="2 deg",
            background_oversampling=pars.get("background_oversampling"),
        )
        dataset = maker.run(obs)
        stacked.stack(dataset)

    counts = stacked.counts
    assert counts.unit == ""
    assert_allclose(counts.data.sum(), pars["counts"], rtol=1e-5)

    exposure = stacked.exposure
    assert exposure.unit == "m2 s"
    assert_allclose(exposure.data.mean(), pars["exposure"], rtol=3e-3)

    background = stacked.background_model.map
    assert background.unit == ""
    assert_allclose(background.data.sum(), pars["background"], rtol=1e-4)

    image_dataset = stacked.to_image()

    counts = image_dataset.counts
    assert counts.unit == ""
    assert_allclose(counts.data.sum(), pars["counts"], rtol=1e-4)

    exposure = image_dataset.exposure
    assert exposure.unit == "m2 s"
    assert_allclose(exposure.data.sum(), pars["exposure_image"], rtol=1e-3)

    background = image_dataset.background_model.map
    assert background.unit == ""
    assert_allclose(background.data.sum(), pars["background"], rtol=1e-4)


@requires_data()
def test_map_maker_ring(observations):
    geomd = geom(ebounds=[0.1, 10])
    map_dataset_maker = MapDatasetMaker(geom=geomd, offset_max="2 deg")
    stacked = MapDatasetOnOff.create(geomd)

    regions = CircleSkyRegion(
        SkyCoord(0, 0, unit="deg", frame="galactic"), radius=0.5 * u.deg
    )
    exclusion = Map.from_geom(geomd)
    exclusion.data = exclusion.geom.region_mask([regions], inside=False)

    ring_bkg = RingBackgroundMaker(
        r_in="0.5 deg", width="0.4 deg", exclusion_mask=exclusion
    )

    for obs in observations:
        dataset = map_dataset_maker.run(obs)
        dataset = dataset.to_image()

        dataset_on_off = ring_bkg.run(dataset)
        stacked.stack(dataset_on_off)

    assert_allclose(np.nansum(stacked.counts.data), 34366, rtol=1e-2)
    assert_allclose(np.nansum(stacked.acceptance_off.data), 434.36, rtol=1e-2)


@requires_data()
def test_map_maker_obs(observations):
    # Test for different spatial geoms and etrue, ereco bins

    geom_reco = geom(ebounds=[0.1, 1, 10])
    e_true = MapAxis.from_edges(
        [0.1, 0.5, 2.5, 10.0], name="energy", unit="TeV", interp="log"
    )
    geom_exp = geom(ebounds=[0.1, 0.5, 2.5, 10.0])
    maker_obs = MapDatasetMaker(
        geom=geom_reco,
        energy_axis_true=e_true,
        binsz_irf=1.0,
        margin_irf=1.0,
        offset_max=2.0 * u.deg,
        cutout=False,
    )

    map_dataset = maker_obs.run(observations[0])
    assert map_dataset.counts.geom == geom_reco
    assert map_dataset.background_model.map.geom == geom_reco
    assert map_dataset.exposure.geom == geom_exp
    assert map_dataset.edisp.edisp_map.data.shape == (3, 48, 6, 11)
    assert map_dataset.edisp.exposure_map.data.shape == (3, 1, 6, 11)
    assert map_dataset.psf.psf_map.data.shape == (3, 66, 6, 11)
    assert map_dataset.psf.exposure_map.data.shape == (3, 1, 6, 11)
    assert_allclose(map_dataset.gti.time_delta, 1800.0 * u.s)
    assert map_dataset.name == "obs_110380"

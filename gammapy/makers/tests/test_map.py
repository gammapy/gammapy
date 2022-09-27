# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from regions import CircleSkyRegion
from gammapy.data import (
    GTI,
    DataStore,
    EventList,
    HDUIndexTable,
    Observation,
    ObservationTable,
)
from gammapy.datasets import MapDataset
from gammapy.datasets.map import RAD_AXIS_DEFAULT
from gammapy.irf import EDispKernelMap, EDispMap, PSFMap
from gammapy.makers import FoVBackgroundMaker, MapDatasetMaker, SafeMaskMaker
from gammapy.maps import HpxGeom, Map, MapAxis, WcsGeom
from gammapy.utils.testing import requires_data, requires_dependency


@pytest.fixture(scope="session")
def observations():
    data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
    obs_id = [110380, 111140]
    return data_store.get_observations(obs_id)


def geom(ebounds, binsz=0.5):
    skydir = SkyCoord(0, -1, unit="deg", frame="galactic")
    energy_axis = MapAxis.from_edges(ebounds, name="energy", unit="TeV", interp="log")
    return WcsGeom.create(
        skydir=skydir, binsz=binsz, width=(10, 5), frame="galactic", axes=[energy_axis]
    )


@pytest.fixture(scope="session")
def geom_config_hpx():
    energy_axis = MapAxis.from_energy_bounds("0.5 TeV", "30 TeV", nbin=3)

    energy_axis_true = MapAxis.from_energy_bounds(
        "0.3 TeV", "30 TeV", nbin=3, per_decade=True, name="energy_true"
    )

    geom_hpx = HpxGeom.create(
        binsz=0.1, frame="galactic", axes=[energy_axis], region="DISK(0, 0, 5.)"
    )
    return {"geom": geom_hpx, "energy_axis_true": energy_axis_true}


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
            "exposure_image": 3.99815e11,
            "background": 27989.05,
            "binsz_irf": 0.5,
            "migra": None,
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
            "migra": None,
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
            "migra": None,
        },
        {
            # Test for different e_true and e_reco bins
            "geom": geom(ebounds=[0.1, 1, 10]),
            "e_true": MapAxis.from_edges(
                [0.1, 0.5, 2.5, 10.0], name="energy_true", unit="TeV", interp="log"
            ),
            "counts": 34366,
            "exposure": 9.951827e08,
            "exposure_image": 5.971096e11,
            "background": 28760.283,
            "background_oversampling": 2,
            "binsz_irf": 0.5,
            "migra": None,
        },
        {
            # Test for different e_true and e_reco and spatial bins
            "geom": geom(ebounds=[0.1, 1, 10]),
            "e_true": MapAxis.from_edges(
                [0.1, 0.5, 2.5, 10.0], name="energy_true", unit="TeV", interp="log"
            ),
            "counts": 34366,
            "exposure": 9.951827e08,
            "exposure_image": 5.971096e11,
            "background": 28760.283,
            "background_oversampling": 2,
            "binsz_irf": 1.0,
            "migra": None,
        },
        {
            # Test for different e_true and e_reco and use edispmap
            "geom": geom(ebounds=[0.1, 1, 10]),
            "e_true": MapAxis.from_edges(
                [0.1, 0.5, 2.5, 10.0], name="energy_true", unit="TeV", interp="log"
            ),
            "counts": 34366,
            "exposure": 9.951827e08,
            "exposure_image": 5.971096e11,
            "background": 28760.283,
            "background_oversampling": 2,
            "binsz_irf": 0.5,
            "migra": MapAxis.from_edges(
                np.linspace(0.0, 3.0, 100), name="migra", unit=""
            ),
        },
    ],
)
def test_map_maker(pars, observations):
    stacked = MapDataset.create(
        geom=pars["geom"],
        energy_axis_true=pars["e_true"],
        binsz_irf=pars["binsz_irf"],
        migra_axis=pars["migra"],
    )

    maker = MapDatasetMaker(background_oversampling=pars.get("background_oversampling"))
    safe_mask_maker = SafeMaskMaker(methods=["offset-max"], offset_max="2 deg")

    for obs in observations:
        cutout = stacked.cutout(position=obs.pointing_radec, width="4 deg")
        dataset = maker.run(cutout, obs)
        dataset = safe_mask_maker.run(dataset, obs)
        stacked.stack(dataset)

    counts = stacked.counts
    assert counts.unit == ""
    assert_allclose(counts.data.sum(), pars["counts"], rtol=1e-5)

    exposure = stacked.exposure
    assert exposure.unit == "m2 s"
    assert_allclose(exposure.data.mean(), pars["exposure"], rtol=3e-3)

    background = stacked.npred_background()
    assert background.unit == ""
    assert_allclose(background.data.sum(), pars["background"], rtol=1e-4)

    image_dataset = stacked.to_image()

    counts = image_dataset.counts
    assert counts.unit == ""
    assert_allclose(counts.data.sum(), pars["counts"], rtol=1e-4)

    exposure = image_dataset.exposure
    assert exposure.unit == "m2 s"
    assert_allclose(exposure.data.sum(), pars["exposure_image"], rtol=1e-3)

    background = image_dataset.npred_background()
    assert background.unit == ""
    assert_allclose(background.data.sum(), pars["background"], rtol=1e-4)


@requires_data()
def test_map_maker_obs(observations):
    # Test for different spatial geoms and etrue, ereco bins

    geom_reco = geom(ebounds=[0.1, 1, 10])
    e_true = MapAxis.from_edges(
        [0.1, 0.5, 2.5, 10.0], name="energy_true", unit="TeV", interp="log"
    )

    reference = MapDataset.create(
        geom=geom_reco, energy_axis_true=e_true, binsz_irf=1.0
    )

    maker_obs = MapDatasetMaker()

    map_dataset = maker_obs.run(reference, observations[0])
    assert map_dataset.counts.geom == geom_reco
    assert map_dataset.npred_background().geom == geom_reco
    assert isinstance(map_dataset.edisp, EDispKernelMap)
    assert map_dataset.edisp.edisp_map.data.shape == (3, 2, 5, 10)
    assert map_dataset.edisp.exposure_map.data.shape == (3, 1, 5, 10)
    assert map_dataset.psf.psf_map.data.shape == (3, 66, 5, 10)
    assert map_dataset.psf.exposure_map.data.shape == (3, 1, 5, 10)
    assert_allclose(map_dataset.gti.time_delta, 1800.0 * u.s)


@requires_data()
def test_map_maker_obs_with_migra(observations):
    # Test for different spatial geoms and etrue, ereco bins
    migra = MapAxis.from_edges(np.linspace(0, 2.0, 50), unit="", name="migra")
    geom_reco = geom(ebounds=[0.1, 1, 10])
    e_true = MapAxis.from_edges(
        [0.1, 0.5, 2.5, 10.0], name="energy_true", unit="TeV", interp="log"
    )

    reference = MapDataset.create(
        geom=geom_reco, energy_axis_true=e_true, migra_axis=migra, binsz_irf=1.0
    )

    maker_obs = MapDatasetMaker()

    map_dataset = maker_obs.run(reference, observations[0])
    assert map_dataset.counts.geom == geom_reco
    assert isinstance(map_dataset.edisp, EDispMap)
    assert map_dataset.edisp.edisp_map.data.shape == (3, 49, 5, 10)
    assert map_dataset.edisp.exposure_map.data.shape == (3, 1, 5, 10)


@requires_data()
def test_make_meta_table(observations):
    maker_obs = MapDatasetMaker()
    map_dataset_meta_table = maker_obs.make_meta_table(observation=observations[0])

    assert_allclose(map_dataset_meta_table["RA_PNT"], 267.68121338)
    assert_allclose(map_dataset_meta_table["DEC_PNT"], -29.6075)
    assert_allclose(map_dataset_meta_table["OBS_ID"], 110380)
    assert map_dataset_meta_table["OBS_MODE"] == "POINTING"


@requires_data()
def test_make_map_no_count(observations):
    dataset = MapDataset.create(geom((0.1, 1, 10)))
    maker_obs = MapDatasetMaker(selection=["exposure"])
    map_dataset = maker_obs.run(dataset, observation=observations[0])

    assert map_dataset.counts is not None
    assert_allclose(map_dataset.counts.data, 0)
    assert map_dataset.counts.geom == dataset.counts.geom


@requires_data()
@requires_dependency("healpy")
def test_map_dataset_maker_hpx(geom_config_hpx, observations):
    reference = MapDataset.create(**geom_config_hpx, binsz_irf=5 * u.deg)

    maker = MapDatasetMaker()
    safe_mask_maker = SafeMaskMaker(
        offset_max="2.5 deg", methods=["aeff-default", "offset-max"]
    )

    dataset = maker.run(reference, observation=observations[0])
    dataset = safe_mask_maker.run(dataset, observation=observations[0]).to_masked()

    assert_allclose(dataset.counts.data.sum(), 4264)
    assert_allclose(dataset.background.data.sum(), 2964.5369, rtol=1e-5)
    assert_allclose(dataset.exposure.data[4, 1000], 5.987e09, rtol=1e-4)

    coords = SkyCoord([0, 3], [0, 0], frame="galactic", unit="deg")
    coords = {"skycoord": coords, "energy": 1 * u.TeV}
    assert_allclose(dataset.mask_safe.get_by_coord(coords), [True, False])

    kernel = dataset.edisp.get_edisp_kernel()

    assert_allclose(kernel.data.sum(axis=1)[3], 1, rtol=0.01)


def test_interpolate_map_dataset():
    energy = MapAxis.from_energy_bounds("1 TeV", "300 TeV", nbin=5, name="energy")
    energy_true = MapAxis.from_nodes(
        np.logspace(-1, 3, 20), name="energy_true", interp="log", unit="TeV"
    )

    # make dummy map IRFs
    geom_allsky = WcsGeom.create(
        npix=(5, 3), proj="CAR", binsz=60, axes=[energy], skydir=(0, 0)
    )
    geom_allsky_true = geom_allsky.drop("energy").to_cube([energy_true])

    # background
    geom_background = WcsGeom.create(
        skydir=(0, 0), width=(5, 5), binsz=0.2 * u.deg, axes=[energy]
    )
    value = 30
    bkg_map = Map.from_geom(geom_background, unit="")
    bkg_map.data = value * np.ones(bkg_map.data.shape)

    # effective area - with a gradient that also depends on energy
    aeff_map = Map.from_geom(geom_allsky_true, unit="cm2 s")
    ra_arr = np.arange(aeff_map.data.shape[1])
    dec_arr = np.arange(aeff_map.data.shape[2])
    for i in np.arange(aeff_map.data.shape[0]):
        aeff_map.data[i, :, :] = (
            (i + 1) * 10 * np.meshgrid(dec_arr, ra_arr)[0]
            + 10 * np.meshgrid(dec_arr, ra_arr)[1]
            + 10
        )
    aeff_map.meta["TELESCOP"] = "HAWC"

    # psf map
    width = 0.2 * u.deg
    rad_axis = MapAxis.from_nodes(np.linspace(0, 2, 50), name="rad", unit="deg")
    psfMap = PSFMap.from_gauss(energy_true, rad_axis, width)

    # edispmap
    edispmap = EDispKernelMap.from_gauss(
        energy, energy_true, sigma=0.1, bias=0.0, geom=geom_allsky
    )

    # events and gti
    nr_ev = 10
    ev_t = Table()
    gti_t = Table()

    ev_t["EVENT_ID"] = np.arange(nr_ev)
    ev_t["TIME"] = nr_ev * [Time("2011-01-01 00:00:00", scale="utc", format="iso")]
    ev_t["RA"] = np.linspace(-1, 1, nr_ev) * u.deg
    ev_t["DEC"] = np.linspace(-1, 1, nr_ev) * u.deg
    ev_t["ENERGY"] = np.logspace(0, 2, nr_ev) * u.TeV

    gti_t["START"] = [Time("2010-12-31 00:00:00", scale="utc", format="iso")]
    gti_t["STOP"] = [Time("2011-01-02 00:00:00", scale="utc", format="iso")]

    events = EventList(ev_t)
    gti = GTI(gti_t)

    # define observation
    obs = Observation(
        obs_id=0,
        obs_info={"RA_PNT": 0.0, "DEC_PNT": 0.0},
        gti=gti,
        aeff=aeff_map,
        edisp=edispmap,
        psf=psfMap,
        bkg=bkg_map,
        events=events,
        obs_filter=None,
    )

    # define analysis geometry
    geom_target = WcsGeom.create(
        skydir=(0, 0), width=(5, 5), binsz=0.1 * u.deg, axes=[energy]
    )

    maker = MapDatasetMaker(
        selection=["exposure", "counts", "background", "edisp", "psf"]
    )
    dataset = MapDataset.create(
        geom=geom_target, energy_axis_true=energy_true, rad_axis=rad_axis, name="test"
    )
    dataset = maker.run(dataset, obs)

    # test counts
    assert dataset.counts.data.sum() == nr_ev

    # test background
    assert np.floor(np.sum(dataset.npred_background().data)) == np.sum(bkg_map.data)
    coords_bg = {"skycoord": SkyCoord("0 deg", "0 deg"), "energy": energy.center[0]}
    assert_allclose(
        dataset.npred_background().get_by_coord(coords_bg)[0], 7.5, atol=1e-4
    )

    # test effective area
    coords_aeff = {
        "skycoord": SkyCoord("0 deg", "0 deg"),
        "energy_true": energy_true.center[0],
    }
    assert_allclose(
        aeff_map.get_by_coord(coords_aeff)[0],
        dataset.exposure.interp_by_coord(coords_aeff)[0],
        atol=1e-3,
    )

    # test edispmap
    pdfmatrix_preinterp = edispmap.get_edisp_kernel(
        position=SkyCoord("0 deg", "0 deg")
    ).pdf_matrix
    pdfmatrix_postinterp = dataset.edisp.get_edisp_kernel(
        position=SkyCoord("0 deg", "0 deg")
    ).pdf_matrix
    assert_allclose(pdfmatrix_preinterp, pdfmatrix_postinterp, atol=1e-7)

    # test psfmap
    geom_psf = geom_target.drop("energy").to_cube([energy_true])
    psfkernel_preinterp = psfMap.get_psf_kernel(
        position=SkyCoord("0 deg", "0 deg"), geom=geom_psf, max_radius=2 * u.deg
    ).data
    psfkernel_postinterp = dataset.psf.get_psf_kernel(
        position=SkyCoord("0 deg", "0 deg"), geom=geom_psf, max_radius=2 * u.deg
    ).data
    assert_allclose(psfkernel_preinterp, psfkernel_postinterp, atol=1e-4)


@requires_data()
@pytest.mark.xfail
def test_minimal_datastore():
    """ "Check that a standard analysis runs on a minimal datastore"""

    energy_axis = MapAxis.from_energy_bounds(
        1, 10, nbin=3, per_decade=False, unit="TeV", name="energy"
    )
    geom = WcsGeom.create(
        skydir=(83.633, 22.014),
        binsz=0.5,
        width=(2, 2),
        frame="icrs",
        proj="CAR",
        axes=[energy_axis],
    )

    data_store = DataStore.from_dir("$GAMMAPY_DATA/tests/minimal_datastore")

    observations = data_store.get_observations()
    maker = MapDatasetMaker()
    offset_max = 2.3 * u.deg
    maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=offset_max)
    circle = CircleSkyRegion(
        center=SkyCoord("83.63 deg", "22.14 deg"), radius=0.2 * u.deg
    )
    exclusion_mask = ~geom.region_mask(regions=[circle])
    maker_fov = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask)

    stacked = MapDataset.create(geom=geom, name="crab-stacked")
    for obs in observations:
        dataset = maker.run(stacked, obs)
        dataset = maker_safe_mask.run(dataset, obs)
        dataset = maker_fov.run(dataset)
        stacked.stack(dataset)

    assert_allclose(stacked.exposure.data.sum(), 6.01909e10)
    assert_allclose(stacked.counts.data.sum(), 1446)
    assert_allclose(stacked.background.data.sum(), 1445.9841)


@requires_data()
def test_dataset_hawc():
    # create the energy reco axis
    energy_axis = MapAxis.from_edges(
        [1.00, 1.78, 3.16, 5.62, 10.0, 17.8, 31.6, 56.2, 100, 177, 316] * u.TeV,
        name="energy",
        interp="log",
    )

    # and energy true axis
    energy_axis_true = MapAxis.from_energy_bounds(
        1e-3, 1e4, nbin=140, unit="TeV", name="energy_true"
    )

    # create a geometry around the Crab location
    geom = WcsGeom.create(
        skydir=SkyCoord(ra=83.63, dec=22.01, unit="deg", frame="icrs"),
        width=3 * u.deg,
        axes=[energy_axis],
        binsz=0.1,
    )

    maker = MapDatasetMaker(
        selection=["counts", "background", "exposure", "edisp", "psf"]
    )
    safemask_maker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

    results = {}
    results["GP"] = [6.53623241669e16, 58, 0.72202391]
    results["NN"] = [6.57154247837e16, 62, 0.76743538]

    for which in ["GP", "NN"]:

        # paths and file names
        data_path = "$GAMMAPY_DATA/hawc/crab_events_pass4/"
        hdu_filename = "hdu-index-table-" + which + "-Crab.fits.gz"
        obs_filename = "obs-index-table-" + which + "-Crab.fits.gz"

        # We want the last event lass for speed
        obs_table = ObservationTable.read(data_path + obs_filename)
        hdu_table = HDUIndexTable.read(data_path + hdu_filename, hdu=9)
        data_store = DataStore(hdu_table=hdu_table, obs_table=obs_table)

        observations = data_store.get_observations()

        # create empty dataset that will contain the data
        geom_exposure = geom.to_image().to_cube([energy_axis_true])
        geom_psf = geom.to_image().to_cube([RAD_AXIS_DEFAULT, energy_axis])
        geom_edisp = geom.to_cube([energy_axis_true])

        dataset_empty = MapDataset.from_geoms(
            geom=geom,
            name="nHit-9",
            geom_exposure=geom_exposure,
            geom_psf=geom_psf,
            geom_edisp=geom_edisp,
        )

        # run the maker
        dataset = maker.run(dataset_empty, observations[0])
        dataset.exposure.meta["livetime"] = "1 s"
        dataset = safemask_maker.run(dataset)

        assert_allclose(dataset.exposure.data.sum(), results[which][0])
        assert_allclose(dataset.counts.data.sum(), results[which][1])
        assert_allclose(dataset.background.data.sum(), results[which][2])

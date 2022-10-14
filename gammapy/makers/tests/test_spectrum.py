# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from regions import CircleSkyRegion
from gammapy.data import DataStore
from gammapy.datasets import SpectrumDataset
from gammapy.makers import (
    ReflectedRegionsBackgroundMaker,
    SafeMaskMaker,
    SpectrumDatasetMaker,
)
from gammapy.maps import MapAxis, RegionGeom, WcsGeom
from gammapy.utils.testing import assert_quantity_allclose, requires_data


@pytest.fixture
def observations_hess_dl3():
    """HESS DL3 observation list."""
    datastore = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    obs_ids = [23523, 23526]
    return datastore.get_observations(obs_ids)


@pytest.fixture
def observations_magic_dl3():
    """MAGIC DL3 observation list."""
    datastore = DataStore.from_dir("$GAMMAPY_DATA/joint-crab/dl3/magic/")
    obs_ids = [5029748]
    return datastore.get_observations(obs_ids, required_irf=["aeff", "edisp"])


@pytest.fixture
def observations_cta_dc1():
    """CTA DC1 observation list."""
    datastore = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
    obs_ids = [110380, 111140]
    return datastore.get_observations(obs_ids)


@pytest.fixture()
def spectrum_dataset_gc():
    e_reco = MapAxis.from_energy_edges(np.logspace(0, 2, 5) * u.TeV, name="energy")
    e_true = MapAxis.from_energy_edges(
        np.logspace(-1, 2, 13) * u.TeV, name="energy_true"
    )
    geom = RegionGeom.create("galactic;circle(0, 0, 0.11)", axes=[e_reco])
    return SpectrumDataset.create(geom=geom, energy_axis_true=e_true)


@pytest.fixture()
def spectrum_dataset_magic_crab():
    e_reco = MapAxis.from_edges(np.logspace(0, 2, 5) * u.TeV, name="energy")
    e_true = MapAxis.from_edges(np.logspace(-0.5, 2, 11) * u.TeV, name="energy_true")
    geom = RegionGeom.create(
        "icrs;circle(83.63, 22.01, 0.14)", axes=[e_reco], binsz_wcs="0.01deg"
    )
    return SpectrumDataset.create(geom=geom, energy_axis_true=e_true)


@pytest.fixture()
def spectrum_dataset_crab():
    e_reco = MapAxis.from_energy_edges(np.logspace(0, 2, 5) * u.TeV)
    e_true = MapAxis.from_energy_edges(
        np.logspace(-0.5, 2, 11) * u.TeV, name="energy_true"
    )
    geom = RegionGeom.create(
        "icrs;circle(83.63, 22.01, 0.11)", axes=[e_reco], binsz_wcs="0.01deg"
    )
    return SpectrumDataset.create(geom=geom, energy_axis_true=e_true)


@pytest.fixture()
def spectrum_dataset_crab_fine():
    e_true = MapAxis.from_energy_edges(
        np.logspace(-2, 2.5, 109) * u.TeV, name="energy_true"
    )
    e_reco = MapAxis.from_energy_edges(np.logspace(-2, 2, 73) * u.TeV)
    geom = RegionGeom.create("icrs;circle(83.63, 22.01, 0.11)", axes=[e_reco])
    return SpectrumDataset.create(geom=geom, energy_axis_true=e_true)


@pytest.fixture
def reflected_regions_bkg_maker():
    pos = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")
    exclusion_region = CircleSkyRegion(pos, Angle(0.3, "deg"))
    geom = WcsGeom.create(skydir=pos, binsz=0.02, width=10.0)
    exclusion_mask = ~geom.region_mask([exclusion_region])

    return ReflectedRegionsBackgroundMaker(
        exclusion_mask=exclusion_mask, min_distance_input="0.2 deg"
    )


@requires_data()
def test_region_center_spectrum_dataset_maker_hess_dl3(
    spectrum_dataset_crab, observations_hess_dl3
):
    datasets = []
    maker = SpectrumDatasetMaker(use_region_center=True)

    for obs in observations_hess_dl3:
        dataset = maker.run(spectrum_dataset_crab, obs)
        datasets.append(dataset)

    assert isinstance(datasets[0], SpectrumDataset)
    assert not datasets[0].exposure.meta["is_pointlike"]

    assert_allclose(datasets[0].counts.data.sum(), 100)
    assert_allclose(datasets[1].counts.data.sum(), 92)

    assert_allclose(datasets[0].exposure.meta["livetime"].value, 1581.736758)
    assert_allclose(datasets[1].exposure.meta["livetime"].value, 1572.686724)

    assert_allclose(datasets[0].npred_background().data.sum(), 7.747881, rtol=1e-5)
    assert_allclose(datasets[1].npred_background().data.sum(), 5.731624, rtol=1e-5)


@requires_data()
def test_spectrum_dataset_maker_hess_dl3(spectrum_dataset_crab, observations_hess_dl3):
    datasets = []
    maker = SpectrumDatasetMaker(use_region_center=False)

    datasets = []
    for obs in observations_hess_dl3:
        dataset = maker.run(spectrum_dataset_crab, obs)
        datasets.append(dataset)

    # Exposure
    assert_allclose(datasets[0].exposure.data.sum(), 7.3111e09)
    assert_allclose(datasets[1].exposure.data.sum(), 6.634534e09)

    # Background
    assert_allclose(datasets[0].npred_background().data.sum(), 7.7429157, rtol=1e-5)
    assert_allclose(datasets[1].npred_background().data.sum(), 5.7314076, rtol=1e-5)

    # Compare background with using bigger region
    e_reco = datasets[0].background.geom.axes["energy"]
    e_true = datasets[0].exposure.geom.axes["energy_true"]
    geom_bigger = RegionGeom.create("icrs;circle(83.63, 22.01, 0.22)", axes=[e_reco])

    datasets_big_region = []
    bigger_region_dataset = SpectrumDataset.create(
        geom=geom_bigger, energy_axis_true=e_true
    )
    for obs in observations_hess_dl3:
        dataset = maker.run(bigger_region_dataset, obs)
        datasets_big_region.append(dataset)

    ratio_regions = (
        datasets[0].counts.geom.solid_angle()
        / datasets_big_region[1].counts.geom.solid_angle()
    )
    ratio_bg_1 = (
        datasets[0].npred_background().data.sum()
        / datasets_big_region[0].npred_background().data.sum()
    )
    ratio_bg_2 = (
        datasets[1].npred_background().data.sum()
        / datasets_big_region[1].npred_background().data.sum()
    )
    assert_allclose(ratio_bg_1, ratio_regions, rtol=1e-2)
    assert_allclose(ratio_bg_2, ratio_regions, rtol=1e-2)

    # Edisp -> it isn't exactly 8, is that right? it also isn't without averaging
    assert_allclose(
        datasets[0].edisp.edisp_map.data[:, :, 0, 0].sum(), e_reco.nbin * 2, rtol=1e-1
    )
    assert_allclose(
        datasets[1].edisp.edisp_map.data[:, :, 0, 0].sum(), e_reco.nbin * 2, rtol=1e-1
    )


@requires_data()
def test_spectrum_dataset_maker_hess_cta(spectrum_dataset_gc, observations_cta_dc1):
    maker = SpectrumDatasetMaker(use_region_center=True)

    datasets = []

    for obs in observations_cta_dc1:
        dataset = maker.run(spectrum_dataset_gc, obs)
        datasets.append(dataset)

    assert_allclose(datasets[0].counts.data.sum(), 53)
    assert_allclose(datasets[1].counts.data.sum(), 47)

    assert_allclose(datasets[0].exposure.meta["livetime"].value, 1764.000034)
    assert_allclose(datasets[1].exposure.meta["livetime"].value, 1764.000034)

    assert_allclose(datasets[0].npred_background().data.sum(), 2.238805, rtol=1e-5)
    assert_allclose(datasets[1].npred_background().data.sum(), 2.165188, rtol=1e-5)


@requires_data()
def test_safe_mask_maker_dl3(spectrum_dataset_crab, observations_hess_dl3):

    safe_mask_maker = SafeMaskMaker(bias_percent=20)
    maker = SpectrumDatasetMaker()

    obs = observations_hess_dl3[0]
    dataset = maker.run(spectrum_dataset_crab, obs)
    dataset = safe_mask_maker.run(dataset, obs)
    assert_allclose(dataset.energy_range[0], 1)
    assert dataset.energy_range[0].unit == "TeV"

    mask_safe = safe_mask_maker.make_mask_energy_aeff_max(dataset)
    assert mask_safe.data.sum() == 4

    mask_safe = safe_mask_maker.make_mask_energy_edisp_bias(dataset)
    assert mask_safe.data.sum() == 3

    mask_safe = safe_mask_maker.make_mask_energy_bkg_peak(dataset)
    assert mask_safe.data.sum() == 4


@requires_data()
def test_safe_mask_maker_dc1(spectrum_dataset_gc, observations_cta_dc1):
    safe_mask_maker = SafeMaskMaker(methods=["edisp-bias", "aeff-max"])

    obs = observations_cta_dc1[0]
    maker = SpectrumDatasetMaker()
    dataset = maker.run(spectrum_dataset_gc, obs)
    dataset = safe_mask_maker.run(dataset, obs)
    assert_allclose(dataset.energy_range[0], 1, rtol=1e-3)
    assert dataset.energy_range[0].unit == "TeV"


@requires_data()
def test_make_meta_table(observations_hess_dl3):
    maker_obs = SpectrumDatasetMaker()
    map_spectrumdataset_meta_table = maker_obs.make_meta_table(
        observation=observations_hess_dl3[0]
    )

    assert_allclose(map_spectrumdataset_meta_table["RA_PNT"], 83.63333129882812)
    assert_allclose(map_spectrumdataset_meta_table["DEC_PNT"], 21.51444435119629)
    assert_allclose(map_spectrumdataset_meta_table["OBS_ID"], 23523)


@requires_data()
def test_region_center_spectrum_dataset_maker_magic_dl3(
    spectrum_dataset_magic_crab, observations_magic_dl3, caplog
):
    maker = SpectrumDatasetMaker(use_region_center=True, selection=["exposure"])
    maker_average = SpectrumDatasetMaker(
        use_region_center=False, selection=["exposure"]
    )
    maker_correction = SpectrumDatasetMaker(
        containment_correction=True, selection=["exposure"]
    )

    # containment correction should fail
    with pytest.raises(ValueError):
        maker_correction.run(spectrum_dataset_magic_crab, observations_magic_dl3[0])

    # use_center = True should run and raise no warning
    dataset = maker.run(spectrum_dataset_magic_crab, observations_magic_dl3[0])

    assert isinstance(dataset, SpectrumDataset)
    assert dataset.exposure.meta["is_pointlike"]
    assert "WARNING" not in [record.levelname for record in caplog.records]

    # use_center = False should raise a warning
    dataset_average = maker_average.run(
        spectrum_dataset_magic_crab, observations_magic_dl3[0]
    )

    assert "WARNING" in [record.levelname for record in caplog.records]
    message = (
        "MapMaker: use_region_center=False should not be used with point-like IRF. "
        "Results are likely inaccurate."
    )
    assert message in [record.message for record in caplog.records]
    assert dataset_average.exposure.meta["is_pointlike"]


@requires_data()
class TestSpectrumMakerChain:
    @staticmethod
    @pytest.mark.parametrize(
        "pars, results",
        [
            (
                dict(containment_correction=False),
                dict(
                    n_on=125, sigma=18.953014, aeff=580254.9 * u.m**2, edisp=0.235864
                ),
            ),
            (
                dict(containment_correction=True),
                dict(
                    n_on=125,
                    sigma=18.953014,
                    aeff=375314.356461 * u.m**2,
                    edisp=0.235864,
                ),
            ),
        ],
    )
    def test_extract(
        pars,
        results,
        observations_hess_dl3,
        spectrum_dataset_crab_fine,
        reflected_regions_bkg_maker,
    ):
        """Test quantitative output for various configs"""
        safe_mask_maker = SafeMaskMaker()
        maker = SpectrumDatasetMaker(
            containment_correction=pars["containment_correction"]
        )

        obs = observations_hess_dl3[0]
        dataset = maker.run(spectrum_dataset_crab_fine, obs)
        dataset = reflected_regions_bkg_maker.run(dataset, obs)
        dataset = safe_mask_maker.run(dataset, obs)

        exposure_actual = (
            dataset.exposure.interp_by_coord(
                {
                    "energy_true": 5 * u.TeV,
                    "skycoord": dataset.counts.geom.center_skydir,
                }
            )
            * dataset.exposure.unit
        )

        edisp_actual = dataset.edisp.get_edisp_kernel().evaluate(
            energy_true=5 * u.TeV, energy=5.2 * u.TeV
        )
        aeff_actual = exposure_actual / dataset.exposure.meta["livetime"]

        assert_quantity_allclose(aeff_actual, results["aeff"], rtol=1e-3)
        assert_quantity_allclose(edisp_actual, results["edisp"], rtol=1e-3)

        # TODO: Introduce assert_stats_allclose
        info = dataset.info_dict()

        assert info["counts"] == results["n_on"]
        assert_allclose(info["sqrt_ts"], results["sigma"], rtol=1e-2)

        gti_obs = obs.gti.table
        gti_dataset = dataset.gti.table
        assert_allclose(gti_dataset["START"], gti_obs["START"])
        assert_allclose(gti_dataset["STOP"], gti_obs["STOP"])

    def test_compute_energy_threshold(
        self, spectrum_dataset_crab_fine, observations_hess_dl3
    ):

        maker = SpectrumDatasetMaker(containment_correction=True)
        safe_mask_maker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

        obs = observations_hess_dl3[0]
        dataset = maker.run(spectrum_dataset_crab_fine, obs)
        dataset = safe_mask_maker.run(dataset, obs)

        actual = dataset.energy_range[0]
        assert_quantity_allclose(actual, 0.681292 * u.TeV, rtol=1e-3)

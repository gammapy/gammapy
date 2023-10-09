from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from gammapy.data import DataStore
from gammapy.datasets import MapDataset
from gammapy.estimators.energydependence import (
    EnergyDependenceEstimator,
    weighted_chi2_parameter,
)
from gammapy.makers import (
    DatasetsMaker,
    FoVBackgroundMaker,
    MapDatasetMaker,
    SafeMaskMaker,
)
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling.models import (
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)


def dataset():
    data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
    obs_id = data_store.obs_table["OBS_ID"][
        data_store.obs_table["OBJECT"] == "MSH 15-5-02"
    ]
    observations = data_store.get_observations(obs_id)

    energy_axis = MapAxis.from_energy_bounds(0.2, 100, nbin=15, unit="TeV")

    source_pos = SkyCoord(320.33, -1.19, unit="deg", frame="galactic")
    geom = WcsGeom.create(
        skydir=(source_pos.galactic.l.deg, source_pos.galactic.b.deg),
        frame="galactic",
        axes=[energy_axis],
        width=5,
        binsz=0.02,
    )
    regions = CircleSkyRegion(center=source_pos, radius=0.7 * u.deg)
    exclusion_mask = geom.region_mask(regions, inside=False)

    safe_mask_maker = SafeMaskMaker(
        methods=["aeff-default", "offset-max"], offset_max=2.5 * u.deg
    )

    dataset_maker = MapDatasetMaker()

    fov_bkg_maker = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask)

    global_dataset = MapDataset.create(geom)
    makers = [dataset_maker, safe_mask_maker, fov_bkg_maker]  # the order matters

    datasets_maker = DatasetsMaker(
        makers, stack_datasets=True, n_jobs=1, cutout_mode="partial"
    )
    datasets = datasets_maker.run(global_dataset, observations)

    return datasets


source_pos = SkyCoord(320.33, -1.19, unit="deg", frame="galactic")

spatial_model = GaussianSpatialModel(
    lon_0=source_pos.l, lat_0=source_pos.b, frame="galactic", sigma="0.1 deg"
)
spectral_model = PowerLawSpectralModel(index=2)
model = SkyModel(
    spatial_model=spatial_model, spectral_model=spectral_model, name="source"
)

model.spatial_model.lon_0.frozen = False
model.spatial_model.lat_0.frozen = False
model.spatial_model.sigma.frozen = False

model.spectral_model.amplitude.frozen = False
model.spectral_model.index.frozen = True

spatial_model.lon_0.min = source_pos.galactic.l.deg - 0.8
spatial_model.lon_0.max = source_pos.galactic.l.deg + 0.8
spatial_model.lat_0.min = source_pos.galactic.b.deg - 0.8
spatial_model.lat_0.max = source_pos.galactic.b.deg + 0.8

datasets = dataset()
datasets.models = model

energy_edges = [0.3, 1, 5, 10] * u.TeV
estimator = EnergyDependenceEstimator(energy_edges=energy_edges, source="source")
results = estimator.run(datasets)


def test_edep():
    results_edep = results["energy_dependence"]["result"]

    assert_allclose(
        results_edep["lon_0"],
        [320.3243, 320.32481, 320.32449, 320.33946] * u.deg,
        atol=1e-13,
    )
    assert_allclose(
        results_edep["lat_0"],
        [-1.2019497, -1.2090126, -1.1939511, -1.2581422] * u.deg,
        atol=1e-13,
    )
    assert_allclose(
        results_edep["sigma"],
        [0.085480195, 0.083817991, 0.088305808, 0.029709042] * u.deg,
        atol=1e-13,
    )
    assert_allclose(
        results["energy_dependence"]["delta_ts"], 11.791874358721543, atol=1e-13
    )


def test_significance():
    results_src = results["src_above_bkg"]

    assert_allclose(
        results_src["delta_ts"],
        [181.91815185037558, 382.6264919312089, 47.9404451529608],
        atol=1e-13,
    )
    assert_allclose(
        results_src["significance"],
        [12.934154684412816, 19.1245726422729, 6.114080695459439],
        atol=1e-13,
    )


def test_chi2():
    results_edep = results["energy_dependence"]["result"]

    chi2_sigma = weighted_chi2_parameter(results_edep, parameter="sigma")
    assert_allclose(chi2_sigma["chi2 sigma"], [15.151767516908023], atol=1e-13)
    assert_allclose(chi2_sigma["significance"], [3.4740491335633923], atol=1e-13)

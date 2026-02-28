# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.data import ObservationTable, HDUIndexTable, DataStore
from gammapy.datasets import Datasets, MapDataset, SpectrumDatasetOnOff
from gammapy.datasets.utils import (
    apply_edisp,
    set_and_restore_mask_fit,
    split_dataset,
    create_global_dataset,
    create_map_dataset_from_dl4,
)
from gammapy.irf import EDispKernel
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling.models import (
    Models,
    PowerLawNormSpectralModel,
    SkyModel,
    TemplateSpatialModel,
)
from gammapy.utils.testing import requires_data


@pytest.fixture
def region_map_true():
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=6, name="energy_true")
    m = Map.create(
        region="icrs;circle(83.63, 21.51, 1)",
        map_type="region",
        axes=[axis],
        unit="1/TeV",
    )
    m.data = np.arange(m.data.size, dtype=float).reshape(m.geom.data_shape)
    return m


def test_apply_edisp(region_map_true):
    e_true = region_map_true.geom.axes[0]
    e_reco = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=3)

    edisp = EDispKernel.from_diagonal_response(
        energy_axis_true=e_true, energy_axis=e_reco
    )

    m = apply_edisp(region_map_true, edisp)
    assert m.geom.data_shape == (3, 1, 1)

    e_reco = m.geom.axes[0].edges
    assert e_reco.unit == "TeV"
    assert m.geom.axes[0].name == "energy"
    assert_allclose(e_reco[[0, -1]].value, [1, 10])


@requires_data()
def test_dataset_split():
    template_diffuse = TemplateSpatialModel.read(
        filename="$GAMMAPY_DATA/fermi-3fhl-gc/gll_iem_v06_gc.fits.gz",
        normalize=False,
    )

    diffuse_iem = SkyModel(
        spectral_model=PowerLawNormSpectralModel(),
        spatial_model=template_diffuse,
        name="diffuse-iem",
    )

    dataset = MapDataset.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc.fits.gz")

    width = 4 * u.deg
    margin = 1 * u.deg

    datasets = split_dataset(dataset, width, margin)
    assert len(datasets) == 15
    assert len(datasets.models) == 0

    datasets = split_dataset(dataset, width, margin, split_template_models=False)
    assert len(datasets.models) == 0

    dataset.models = Models()
    datasets = split_dataset(dataset, width, margin)
    assert len(datasets.models) == 0

    dataset.models = Models([diffuse_iem])

    datasets = split_dataset(dataset, width, margin, split_template_models=False)
    assert len(datasets) == 15
    assert len(datasets.models) == 1
    assert datasets.models[0].name == "diffuse-iem"

    datasets = split_dataset(
        dataset, width=width, margin=margin, split_template_models=True
    )
    assert len(datasets.models) == len(datasets)
    assert len(datasets.parameters.free_parameters) == 1
    assert "diffuse-iem" in datasets.models[0].name
    assert (
        datasets[7].models[0].spatial_model.map.geom.width[0][0] == width + 2 * margin
    )
    assert (
        datasets[7].models[0].spatial_model.map.geom.width[1][0] == width + 2 * margin
    )

    geom = dataset.counts.geom
    pixel_width = np.ceil((width / geom.pixel_scales).to_value("")).astype(int)
    margin_width = np.ceil((margin / geom.pixel_scales).to_value("")).astype(int)

    assert datasets[7].mask_fit.data[0, :, :].sum() == np.prod(pixel_width)
    assert (~datasets[7].mask_fit.data[0, :, :]).sum() == np.prod(
        pixel_width + 2 * margin_width
    ) - np.prod(pixel_width)


@requires_data()
def test_create_global_dataset():
    base_dataset = MapDataset.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc.fits.gz")
    width = 4 * u.deg
    margin = 1 * u.deg
    datasets = split_dataset(base_dataset, width, margin)

    global_dataset = create_global_dataset(datasets)
    assert global_dataset.counts.geom.width.max() > base_dataset.counts.geom.width.max()
    assert (
        global_dataset.counts.geom.axes[0].nbin > base_dataset.counts.geom.axes[0].nbin
    )
    assert_allclose(
        global_dataset.counts.geom.axes[0].edges.min(),
        base_dataset.counts.geom.axes[0].edges.min(),
    )
    assert_allclose(
        global_dataset.counts.geom.axes[0].edges.max(),
        base_dataset.counts.geom.axes[0].edges.max(),
    )
    assert_allclose(
        global_dataset.exposure.geom.axes[0].edges.min(),
        base_dataset.exposure.geom.axes[0].edges.min(),
    )
    assert_allclose(
        global_dataset.exposure.geom.axes[0].edges.max(),
        base_dataset.exposure.geom.axes[0].edges.max(),
    )

    global_dataset = create_global_dataset(
        datasets,
        width=(20, 10) * u.deg,
        binsz=0.02 * u.deg,
        name="test",
        position=datasets[0].counts.geom.center_skydir,
        energy_min=0.05 * u.TeV,
        energy_max=0.2 * u.TeV,
        energy_true_min=0.02 * u.TeV,
        energy_true_max=0.4 * u.TeV,
        nbin_per_decade=10,
    )
    assert global_dataset.name == "test"
    assert_allclose(global_dataset.counts.geom.width.max(), 20 * u.deg)
    assert_allclose(global_dataset.counts.geom.pixel_scales.max(), 0.02 * u.deg)
    assert_allclose(global_dataset.counts.geom.axes[0].nbin, 7)
    assert_allclose(
        global_dataset.counts.geom.center_skydir.separation(
            datasets[0].counts.geom.center_skydir
        ).value,
        0,
        atol=1e-5,
    )
    assert_allclose(global_dataset.counts.geom.axes[0].edges.min(), 0.05 * u.TeV)
    assert_allclose(global_dataset.counts.geom.axes[0].edges.max(), 0.2 * u.TeV)
    assert_allclose(global_dataset.exposure.geom.axes[0].edges.min(), 0.02 * u.TeV)
    assert_allclose(global_dataset.exposure.geom.axes[0].edges.max(), 0.4 * u.TeV)


@requires_data()
def test_set_and_restore_mask():
    ds1 = SpectrumDatasetOnOff.read(
        "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits"
    )
    ds2 = SpectrumDatasetOnOff.read(
        "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23526.fits"
    )

    datasets = Datasets([ds1, ds2])
    with set_and_restore_mask_fit(
        datasets, None, 800 * u.GeV, 5 * u.TeV, round_to_edge=True
    ) as masked_datasets:
        range1 = masked_datasets[0].energy_range_fit
        range2 = masked_datasets[1].energy_range_fit
        assert not np.all(masked_datasets[0].mask_fit)
        assert not np.all(masked_datasets[1].mask_fit)

    assert_allclose(range1[0].quantity.to_value("TeV"), 0.7943282)
    assert_allclose(range1[1].quantity.to_value("TeV"), 5.011872)
    assert_allclose(range2[0].quantity.to_value("TeV"), 0.7943282)

    range1 = datasets[0].energy_range_fit
    range2 = datasets[1].energy_range_fit
    assert_allclose(range1[0].quantity.to_value("TeV"), 0.01)
    assert_allclose(range1[1].quantity.to_value("TeV"), 100.0)
    assert_allclose(range2[0].quantity.to_value("TeV"), 0.01)

    with set_and_restore_mask_fit(
        datasets, None, 100 * u.TeV, 200 * u.TeV
    ) as masked_datasets:
        assert len(masked_datasets) == 0

    with set_and_restore_mask_fit(datasets, None) as masked_datasets:
        assert np.all(masked_datasets[0].mask_fit)
        assert np.all(masked_datasets[1].mask_fit)

    mask = ds1.counts.geom.energy_mask(800 * u.GeV, 5 * u.TeV, round_to_edge=True)
    datasets = Datasets([ds1, ds2])
    for d in datasets:
        d.mask_fit = mask

    with set_and_restore_mask_fit(datasets) as masked_datasets:
        assert_allclose(masked_datasets[0].mask_fit, mask)
        assert not np.all(masked_datasets[0].mask_fit)
        assert not np.all(masked_datasets[1].mask_fit)

    with set_and_restore_mask_fit(datasets, operator=None) as masked_datasets:
        assert masked_datasets[0].mask_fit == mask
        assert np.all(masked_datasets[0].mask_fit)
        assert np.all(masked_datasets[1].mask_fit)


@requires_data()
def test_set_and_restore_mask_3d():
    dataset = MapDataset.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc.fits.gz")

    mask_fit = Map.from_geom(dataset.counts.geom, data=True)
    mask_fit = mask_fit.binary_erode(2 * u.deg)

    datasets = Datasets([dataset])
    with set_and_restore_mask_fit(
        datasets, mask_fit, 80 * u.GeV, 500 * u.GeV, round_to_edge=True
    ) as masked_datasets:
        range1 = masked_datasets[0].energy_range_fit

    assert_allclose(range1[0].quantity[100, 220].to_value("GeV"), 84.47164)
    assert_allclose(range1[1].quantity[100, 220].to_value("GeV"), 500.0)
    assert np.isnan(range1[0].quantity[0, 0].to_value("GeV"))
    assert np.isnan(range1[1].quantity[0, 0].to_value("GeV"))

    range1 = datasets[0].energy_range_fit
    assert_allclose(range1[0].quantity[100, 220].to_value("GeV"), 10.0)
    assert_allclose(range1[1].quantity[100, 220].to_value("GeV"), 500.0)
    assert_allclose(range1[0].quantity[0, 0].to_value("GeV"), 10.0)
    assert_allclose(range1[1].quantity[0, 0].to_value("GeV"), 500.0)

    for d in datasets:
        d.mask_fit = mask_fit

    mask_fit_all_true = Map.from_geom(dataset.counts.geom, data=True)

    with set_and_restore_mask_fit(
        datasets, mask_fit=mask_fit_all_true
    ) as masked_datasets:
        assert not np.all(masked_datasets[0].mask_fit)

    with set_and_restore_mask_fit(
        datasets, mask_fit=mask_fit_all_true, operator=None
    ) as masked_datasets:
        assert np.all(masked_datasets[0].mask_fit)


@requires_data()
def test_create_map_dataset_from_dl4_hawc():
    energy_estimator = "NN"
    data_path = "$GAMMAPY_DATA/hawc/crab_events_pass4/"
    hdu_filename = f"hdu-index-table-{energy_estimator}-Crab.fits.gz"
    obs_filename = f"obs-index-table-{energy_estimator}-Crab.fits.gz"
    obs_table = ObservationTable.read(data_path + obs_filename)

    fhit = 6
    hdu_table = HDUIndexTable.read(data_path + hdu_filename, hdu=fhit)
    data_store = DataStore(hdu_table=hdu_table, obs_table=obs_table)
    obs = data_store.get_observations()[0]

    energy_axis = MapAxis.from_edges(
        [1.00, 1.78, 3.16, 5.62, 10.0, 17.8, 31.6, 56.2, 100, 177, 316] * u.TeV,
        name="energy",
        interp="log",
    )

    energy_axis_true = MapAxis.from_energy_bounds(
        1e-3, 1e4, nbin=140, unit="TeV", name="energy_true"
    )

    geom = WcsGeom.create(
        skydir=SkyCoord(ra=83.63, dec=22.01, unit="deg", frame="icrs"),
        width=6 * u.deg,
        axes=[energy_axis],
        binsz=0.05,
    )

    dataset = create_map_dataset_from_dl4(
        obs, geom=geom, energy_axis_true=energy_axis_true, name=f"fHit {fhit}"
    )

    assert dataset.psf.energy_name == "energy"
    assert dataset.psf.psf_map.geom.to_image() == obs.psf.psf_map.geom.to_image()
    assert dataset.psf.psf_map.geom.axes["rad"] == obs.psf.psf_map.geom.axes["rad"]
    assert (
        dataset.psf.psf_map.geom.axes["energy"] != obs.psf.psf_map.geom.axes["energy"]
    )

    assert (
        dataset.edisp.edisp_map.geom.to_image() == obs.edisp.edisp_map.geom.to_image()
    )
    assert dataset.edisp.edisp_map.geom != obs.edisp.edisp_map.geom

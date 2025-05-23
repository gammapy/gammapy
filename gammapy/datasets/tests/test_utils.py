# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.datasets import MapDataset
from gammapy.datasets.utils import apply_edisp, split_dataset, create_global_dataset
from gammapy.irf import EDispKernel
from gammapy.maps import Map, MapAxis
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

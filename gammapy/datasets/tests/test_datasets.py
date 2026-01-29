# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import pytest
import numpy as np
import astropy.units as u
from numpy.testing import assert_allclose, assert_equal
from gammapy.datasets import Datasets, SpectrumDatasetOnOff
from gammapy.datasets.tests.test_map import get_map_dataset
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    FoVBackgroundModel,
    Models,
    SkyModel,
    PiecewiseNormSpectralModel,
)
from gammapy.modeling.tests.test_fit import MyDataset
from gammapy.stats import GaussianPriorPenalty
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def datasets():
    return Datasets([MyDataset(name="test-1"), MyDataset(name="test-2")])


@pytest.fixture(scope="session")
def map_datasets():
    axis = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=2)
    geom = WcsGeom.create(
        skydir=(266.40498829, -28.93617776),
        binsz=0.05,
        width=(20, 20),
        frame="icrs",
        axes=[axis],
    )

    axis = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=3, name="energy_true")
    geom_etrue = WcsGeom.create(
        skydir=(266.40498829, -28.93617776),
        binsz=0.05,
        width=(20, 20),
        frame="icrs",
        axes=[axis],
    )

    dataset_1 = get_map_dataset(geom, geom_etrue, name="test-1")
    dataset_1.mask_fit = None
    dataset_1.background /= 400
    dataset_2 = get_map_dataset(geom, geom_etrue, name="test-2")
    datasets = Datasets([dataset_1, dataset_2])

    model = SkyModel.create("pl", "point", name="src")
    model.spatial_model.position = dataset_1.exposure.geom.center_skydir

    model2 = model.copy()
    model2.spatial_model.lon_0.value += 0.1
    model2.spatial_model.lat_0.value += 0.1

    models = Models(
        [
            model,
            model2,
            FoVBackgroundModel(dataset_name=dataset_1.name),
            FoVBackgroundModel(dataset_name=dataset_2.name),
        ]
    )

    datasets.models = models
    dataset_1.fake(random_state=42)
    dataset_2.fake(random_state=42)
    map_datasets = Datasets([dataset_1, dataset_2])
    return map_datasets


def test_datasets_init(datasets):
    # Passing a Python list of `Dataset` objects should work
    Datasets(list(datasets))

    # Passing an existing `Datasets` object should work
    Datasets(datasets)


def test_datasets_types(datasets):
    assert datasets.is_all_same_type


def test_datasets_likelihood(datasets):
    likelihood = datasets.stat_sum()
    assert_allclose(likelihood, 14472200.0002)


@requires_data()
def test_datasets_likelihood_with_penalty(map_datasets):
    assert_allclose(map_datasets.stat_sum(), 4132.493313)

    norm_model = PiecewiseNormSpectralModel(energy=np.geomspace(0.1, 10, 5) * u.TeV)

    penalty = GaussianPriorPenalty.from_method(
        norm_model.parameters, "L2", mean=0.0, lambda_=2
    )

    map_datasets.models["src"].spectral_model *= norm_model

    models = Models(map_datasets.models, penalties=penalty)
    assert len(models._penalties) == 1
    models.set_penalties(penalty)
    assert len(models._penalties) == 1

    assert_allclose(penalty.stat_sum(), 10)

    map_datasets.models = models

    assert_allclose(map_datasets.stat_sum(), 4132.493313 + 10)

    map_datasets.models["src"].spectral_model = map_datasets.models[
        "src"
    ].spectral_model.model1
    models = Models(map_datasets.models, penalties=None)
    assert models._penalties == map_datasets.models._penalties
    models.set_penalties(None)
    assert models._penalties is None
    map_datasets.models = (
        models  # TODO: removing the penalty is quite complex should be simpler
    )

    assert_allclose(map_datasets.stat_sum(), 4132.493313)

    with pytest.raises(ValueError):
        models.set_penalties([[1, 2]])


def test_datasets_str(datasets):
    assert "Datasets" in str(datasets)


def test_datasets_getitem(datasets):
    assert datasets["test-1"].name == "test-1"
    assert datasets["test-2"].name == "test-2"


def test_names(datasets):
    assert datasets.names == ["test-1", "test-2"]


def test_datasets_mutation():
    dat = MyDataset(name="test-1")
    dats = Datasets([MyDataset(name="test-2"), MyDataset(name="test-3")])
    dats2 = Datasets([MyDataset(name="test-4"), MyDataset(name="test-5")])

    dats.insert(0, dat)
    assert dats.names == ["test-1", "test-2", "test-3"]

    dats.extend(dats2)
    assert dats.names == ["test-1", "test-2", "test-3", "test-4", "test-5"]

    dat3 = dats[3]
    dats.remove(dats[3])
    assert dats.names == ["test-1", "test-2", "test-3", "test-5"]
    dats.append(dat3)
    assert dats.names == ["test-1", "test-2", "test-3", "test-5", "test-4"]
    dats.pop(3)
    assert dats.names == ["test-1", "test-2", "test-3", "test-4"]

    with pytest.raises(ValueError, match="Dataset names must be unique"):
        dats.append(dat)

    with pytest.raises(ValueError, match="Dataset names must be unique"):
        dats.insert(0, dat)

    with pytest.raises(ValueError, match="Dataset names must be unique"):
        dats.extend(dats2)


@requires_data()
def test_datasets_info_table():
    datasets_hess = Datasets()

    for obs_id in [23523, 23526]:
        dataset = SpectrumDatasetOnOff.read(
            f"$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{obs_id}.fits"
        )
        datasets_hess.append(dataset)

    table = datasets_hess.info_table()
    assert table["name"][0] == "23523"
    assert table["name"][1] == "23526"
    assert np.isnan(table["npred_signal"][0])
    assert_equal(table["counts"], [124, 126])
    assert_allclose(table["background"], [7.666, 8.583], rtol=1e-3)

    table_cumul = datasets_hess.info_table(cumulative=True)
    assert table_cumul["name"][0] == "stacked"
    assert table_cumul["name"][1] == "stacked"
    assert np.isnan(table_cumul["npred_signal"][0])
    assert table_cumul["alpha"][1] == table_cumul["alpha"][0]
    assert_equal(table_cumul["counts"], [124, 250])
    assert_allclose(table_cumul["background"], [7.666, 16.25], rtol=1e-3)

    assert table["excess"].sum() == table_cumul["excess"][1]
    assert table["counts"].sum() == table_cumul["counts"][1]
    assert table["background"].sum() == table_cumul["background"][1]

    datasets_hess[0].mask_safe.data = ~datasets_hess[0].mask_safe.data
    assert datasets_hess.info_table()["counts"][0] == 65

    datasets_hess[0].mask_safe.data = np.ones_like(
        datasets_hess[0].mask_safe.data, dtype=bool
    )
    datasets_hess[1].mask_safe.data = np.ones_like(
        datasets_hess[1].mask_safe.data, dtype=bool
    )
    assert_equal(datasets_hess.info_table()["counts"], [189, 199])
    assert_equal(datasets_hess.info_table(cumulative=True)["counts"], [189, 388])


@requires_data()
def test_datasets_write(map_datasets, tmp_path):
    datasets = Datasets(map_datasets[0])

    datasets.write(
        filename=tmp_path / "test",
        filename_models=tmp_path / "test_model",
        overwrite=False,
    )
    os.remove(tmp_path / "test-1.fits")

    with pytest.raises(OSError):
        datasets.write(
            filename=tmp_path / "test",
            filename_models=tmp_path / "test_model",
            overwrite=False,
        )


@requires_data()
def test_datasets_fit(map_datasets):
    datasets = map_datasets

    fit = Fit()
    results = fit.run(datasets)

    assert_allclose(results.models.covariance.data, datasets.models.covariance.data)


def test_add_datasets(datasets):
    assert isinstance(datasets + Datasets(), Datasets)
    assert isinstance(Datasets() + datasets[0], Datasets)
    assert isinstance(datasets[0] + Datasets(), Datasets)
    assert isinstance(datasets[0] + datasets[1], Datasets)
    assert isinstance(datasets[0] + [], Datasets)

    assert len(datasets[0] + datasets[1]) == 2

    new_datasets = datasets[0] + datasets[1]
    assert len(new_datasets) == len(datasets.models)
    assert len(new_datasets) == 2

    tmp = np.ones(3)
    with pytest.raises(TypeError):
        datasets[0] + tmp

    with pytest.raises(TypeError):
        datasets + tmp

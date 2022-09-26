# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
import astropy.units as u
from regions import CircleSkyRegion
from gammapy.datasets import Datasets, SpectrumDataset
from gammapy.modeling.models import DatasetModels
from gammapy.utils.testing import requires_data


@requires_data()
def test_datasets_to_io(tmp_path):
    filedata = "$GAMMAPY_DATA/tests/models/gc_example_datasets.yaml"
    filemodel = "$GAMMAPY_DATA/tests/models/gc_example_models.yaml"

    datasets = Datasets.read(
        filename=filedata,
        filename_models=filemodel,
    )

    assert len(datasets) == 2
    assert len(datasets.models) == 5
    dataset0 = datasets[0]
    assert dataset0.name == "gc"
    assert dataset0.counts.data.sum() == 22258
    assert_allclose(dataset0.exposure.data.sum(), 8.057342e12, atol=0.1)
    assert dataset0.psf is not None
    assert dataset0.edisp is not None

    assert_allclose(dataset0.npred_background().data.sum(), 15726.8, atol=0.1)

    assert dataset0.background_model.name == "gc-bkg"

    dataset1 = datasets[1]
    assert dataset1.name == "g09"
    assert dataset1.background_model.name == "g09-bkg"

    assert (
        dataset0.models["gll_iem_v06_cutout"] == dataset1.models["gll_iem_v06_cutout"]
    )

    assert isinstance(dataset0.models, DatasetModels)
    assert len(dataset0.models) == 4
    assert dataset0.models[0].name == "gc"
    assert dataset0.models[1].name == "gll_iem_v06_cutout"
    assert dataset0.models[2].name == "gc-bkg"

    assert (
        dataset0.models["gc"].parameters["reference"]
        is dataset1.models["g09"].parameters["reference"]
    )
    assert_allclose(dataset1.models["g09"].parameters["lon_0"].value, 0.9, atol=0.1)

    datasets.write(
        filename=tmp_path / "written_datasets.yaml",
        filename_models=tmp_path / "written_models.yaml",
    )

    datasets_read = Datasets.read(
        filename=tmp_path / "written_datasets.yaml",
        filename_models=tmp_path / "written_models.yaml",
    )

    assert len(datasets.parameters) == 22

    assert len(datasets_read) == 2
    dataset0 = datasets_read[0]
    assert dataset0.name == "gc"
    assert dataset0.counts.data.sum() == 22258
    assert_allclose(dataset0.exposure.data.sum(), 8.057342e12, atol=0.1)
    assert dataset0.psf is not None
    assert dataset0.edisp is not None
    assert_allclose(dataset0.npred_background().data.sum(), 15726.8, atol=0.1)
    assert datasets[1].name == "g09"

    dataset_copy = dataset0.copy(name="dataset0-copy")
    assert dataset_copy.models is None


@requires_data()
def test_spectrum_datasets_to_io(tmp_path):
    filedata = "$GAMMAPY_DATA/tests/models/gc_example_datasets.yaml"
    filemodel = "$GAMMAPY_DATA/tests/models/gc_example_models.yaml"

    datasets = Datasets.read(
        filename=filedata,
        filename_models=filemodel,
    )
    reg = CircleSkyRegion(center=datasets[0]._geom.center_skydir, radius=1.0 * u.deg)
    dataset0 = datasets[0].to_spectrum_dataset(reg)
    datasets1 = Datasets([dataset0, datasets[1]])
    datasets1.write(
        filename=tmp_path / "written_datasets.yaml",
        filename_models=tmp_path / "written_models.yaml",
    )

    datasets_read = Datasets.read(
        filename=tmp_path / "written_datasets.yaml",
        filename_models=tmp_path / "written_models.yaml",
    )

    assert len(datasets_read.parameters) == 19

    assert len(datasets_read) == 2

    assert datasets_read[0].counts.data.sum() == 18429
    assert_allclose(datasets_read[0].exposure.data.sum(), 2.034089e10, atol=0.1)
    assert isinstance(datasets_read[0], SpectrumDataset)

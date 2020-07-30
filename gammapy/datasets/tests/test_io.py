# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from gammapy.datasets import Datasets
from gammapy.modeling.models import Models
from gammapy.utils.testing import requires_data, requires_dependency
from gammapy.modeling import Fit


@requires_data()
@requires_dependency("iminuit")
def test_datasets_to_io(tmp_path):
    path = "$GAMMAPY_DATA/tests/models"
    filedata = "gc_example_datasets.yaml"
    filemodel = "gc_example_models.yaml"

    datasets = Datasets.read(path, filedata, filemodel)

    assert len(datasets) == 2
    print(list(datasets.models))
    assert len(datasets.models) == 5
    dataset0 = datasets[0]
    assert dataset0.name == "gc"
    assert dataset0.counts.data.sum() == 22258
    assert_allclose(dataset0.exposure.data.sum(), 8.057342e+12, atol=0.1)
    assert dataset0.psf is not None
    assert dataset0.edisp is not None

    assert_allclose(dataset0.background_model.evaluate().data.sum(), 15726.8, atol=0.1)

    assert dataset0.background_model.name == "gc-bkg"

    dataset1 = datasets[1]
    assert dataset1.name == "g09"
    assert dataset1.background_model.name == "g09-bkg"

    assert (
        dataset0.models["gll_iem_v06_cutout"] == dataset1.models["gll_iem_v06_cutout"]
    )

    assert isinstance(dataset0.models, Models)
    assert len(dataset0.models) ==4
    assert dataset0.models[0].name == "gc"
    assert dataset0.models[1].name == "gll_iem_v06_cutout"
    assert dataset0.models[2].name == "gc-bkg"

    assert (
        dataset0.models["gc"].parameters["reference"]
        is dataset1.models["g09"].parameters["reference"]
    )
    assert_allclose(dataset1.models["g09"].parameters["lon_0"].value, 0.9, atol=0.1)

    datasets.write(tmp_path, prefix="written")
    datasets_read = Datasets.read(
        tmp_path, "written_datasets.yaml", "written_models.yaml"
    )

    assert len(datasets.parameters) == 22

    assert len(datasets_read) == 2
    dataset0 = datasets_read[0]
    assert dataset0.counts.data.sum() == 22258
    assert_allclose(dataset0.exposure.data.sum(), 8.057342e+12, atol=0.1)
    assert dataset0.psf is not None
    assert dataset0.edisp is not None
    assert_allclose(dataset0.background_model.evaluate().data.sum(), 15726.8, atol=0.1)

    dataset_copy = dataset0.copy(name="dataset0-copy")
    assert dataset_copy.background_model.datasets_names == ["dataset0-copy"]

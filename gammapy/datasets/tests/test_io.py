# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from gammapy.datasets import Datasets
from gammapy.modeling.models import Models
from gammapy.utils.testing import requires_data
from gammapy.modeling import Fit


@requires_data()
def test_datasets_to_io(tmp_path):
    filedata = "$GAMMAPY_DATA/tests/models/gc_example_datasets.yaml"
    filemodel = "$GAMMAPY_DATA/tests/models/gc_example_models.yaml"

    datasets = Datasets.read(filedata, filemodel)

    assert len(datasets) == 2

    dataset0 = datasets[0]
    assert dataset0.name == "gc"
    assert dataset0.counts.data.sum() == 6824
    assert_allclose(dataset0.exposure.data.sum(), 2072125400000.0, atol=0.1)
    assert dataset0.psf is not None
    assert dataset0.edisp is not None

    assert_allclose(dataset0.background_model.evaluate().data.sum(), 4094.2, atol=0.1)

    assert dataset0.background_model.name == "background_irf_gc"

    dataset1 = datasets[1]
    assert dataset1.name == "g09"
    assert dataset1.background_model.name == "background_irf_g09"

    assert (
        dataset0.models["gll_iem_v06_cutout"] == dataset1.models["gll_iem_v06_cutout"]
    )

    assert isinstance(dataset0.models, Models)
    assert len(dataset0.models) == 5
    assert dataset0.models[0].name == "gc"
    assert dataset0.models[1].name == "gll_iem_v06_cutout"

    assert (
        dataset0.models["background_irf_gc"].parameters["norm"]
        is dataset1.models["background_irf_g09"].parameters["norm"]
    )

    assert (
        dataset0.models["gc"].parameters["reference"]
        is dataset1.models["g09"].parameters["reference"]
    )
    assert_allclose(dataset1.models["g09"].parameters["lon_0"].value, 0.9, atol=0.1)

    datasets.write(tmp_path, prefix="written")
    datasets_read = Datasets.read(
        tmp_path / "written_datasets.yaml", tmp_path / "written_models.yaml"
    )

    assert len(datasets.parameters) == 21

    assert len(datasets_read) == 2
    dataset0 = datasets_read[0]
    assert dataset0.counts.data.sum() == 6824
    assert_allclose(dataset0.exposure.data.sum(), 2072125400000.0, atol=0.1)
    assert dataset0.psf is not None
    assert dataset0.edisp is not None
    assert_allclose(dataset0.background_model.evaluate().data.sum(), 4094.2, atol=0.1)

    Fit(datasets).run()
    assert_allclose(
        datasets.models["background_irf_g09"].covariance,
        datasets.models["background_irf_gc"].covariance,
    )
from gammapy.utils.testing import requires_data, requires_dependency
from numpy.testing import assert_allclose
from gammapy.modeling.models import SkyModel
from gammapy.datasets import Datasets, SpectrumDatasetOnOff
from gammapy.modeling.sampler import Sampler
from gammapy.modeling.models import (
    UniformPrior,
    LogUniformPrior,
)


@requires_dependency("ultranest")
@requires_data()
def test_run(backend="ultranest"):
    datasets = Datasets()
    for obs_id in [23523, 23526]:
        dataset = SpectrumDatasetOnOff.read(
            f"$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{obs_id}.fits"
        )
        datasets.append(dataset)

    datasets.models = [SkyModel.create(spectral_model="pl")]
    datasets.models.parameters["index"].prior = UniformPrior(min=2, max=3)
    datasets.models.parameters["amplitude"].prior = LogUniformPrior(
        min=1e-12, max=1e-10
    )

    sampler_opts = {"live_points": 300}
    sampler = Sampler(backend=backend, sampler_opts=sampler_opts)

    result = sampler.run(datasets)

    assert result.success
    assert sampler._sampler.min_num_live_points == sampler_opts["live_points"]
    assert (
        result.samples.shape[1]
        == datasets.models.parameters.free_parameters.value.shape[0]
    )

    required_keys = [
        "logz",
        "logzerr",
        "posterior",
        "samples",
        "ncall",
        "insertion_order_MWW_test",
    ]
    assert set(required_keys).issubset(result.sampler_results.keys())

    assert_allclose(result.models.parameters["index"].value, 2.6, rtol=0.1)
    assert_allclose(result.models.parameters["amplitude"].value, 4e-11, rtol=0.2)

    assert result.models.parameters["index"].error > 0
    assert result.models.parameters["amplitude"].error > 0
    assert result.models._covariance is None

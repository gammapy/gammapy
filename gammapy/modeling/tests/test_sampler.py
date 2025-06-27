from gammapy.utils.testing import requires_data, requires_dependency
from numpy.testing import assert_allclose
from gammapy.modeling.models import SkyModel
from gammapy.datasets import Datasets, SpectrumDatasetOnOff
from gammapy.modeling.sampler import Sampler
from gammapy.modeling.models import (
    UniformPrior,
    LogUniformPrior,
    PowerLawSpectralModel,
    Models,
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

    pwl1 = PowerLawSpectralModel(index=2.3)
    pwl1.amplitude.prior = LogUniformPrior(min=1e-12, max=1e-10)
    pwl1.index.prior = UniformPrior(min=2, max=3)

    models = Models([SkyModel(pwl1, name="source1")])
    datasets.models = models

    sampler_opts = {"live_points": 300}
    sampler = Sampler(backend=backend, sampler_opts=sampler_opts)

    result = sampler.run(datasets)

    assert result.success
    assert sampler._sampler.min_num_live_points == sampler_opts["live_points"]
    assert (
        result.samples.shape[1]
        == datasets.models.parameters.free_unique_parameters.value.shape[0]
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

    assert (
        result.models.parameters["index"].value
        == result.sampler_results["posterior"]["mean"][0]
    )
    assert (
        result.models.parameters["amplitude"].value
        == result.sampler_results["posterior"]["mean"][1]
    )
    assert (
        result.models.parameters["index"].error
        == result.sampler_results["posterior"]["stdev"][0]
    )
    assert (
        result.models.parameters["amplitude"].error
        == result.sampler_results["posterior"]["stdev"][1]
    )

    assert_allclose(result.models.parameters["index"].value, 2.7, rtol=0.1)
    assert_allclose(result.models.parameters["amplitude"].value, 4e-11, rtol=0.1)
    assert_allclose(result.models.parameters["index"].error, 0.1, rtol=0.2)
    assert_allclose(result.models.parameters["amplitude"].error, 3.2e-12, rtol=0.2)

    assert result.models._covariance is None


@requires_dependency("ultranest")
@requires_data()
def test_run_linked_params(backend="ultranest"):
    datasets = Datasets()
    for obs_id in [23523, 23526]:
        dataset = SpectrumDatasetOnOff.read(
            f"$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{obs_id}.fits"
        )
        datasets.append(dataset)

    # test with linked parameters
    pwl1 = PowerLawSpectralModel(index=2.3)
    pwl1.amplitude.prior = LogUniformPrior(min=1e-12, max=1e-10)
    pwl1.index.prior = UniformPrior(min=2, max=3)

    pwl2 = PowerLawSpectralModel()
    pwl2.index = pwl1.index
    pwl2.amplitude = pwl1.amplitude

    models = Models([SkyModel(pwl1, name="source1"), SkyModel(pwl2, name="source2")])
    datasets.models = models

    sampler_opts = {"live_points": 300}
    sampler = Sampler(backend=backend, sampler_opts=sampler_opts)

    result = sampler.run(datasets)

    assert result.success
    assert sampler._sampler.min_num_live_points == sampler_opts["live_points"]
    assert (
        result.samples.shape[1]
        == datasets.models.parameters.free_unique_parameters.value.shape[0]
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

    assert (
        result.models.parameters["index"].value
        == result.sampler_results["posterior"]["mean"][0]
    )
    assert (
        result.models.parameters["amplitude"].value
        == result.sampler_results["posterior"]["mean"][1]
    )
    assert (
        result.models.parameters["index"].error
        == result.sampler_results["posterior"]["stdev"][0]
    )
    assert (
        result.models.parameters["amplitude"].error
        == result.sampler_results["posterior"]["stdev"][1]
    )

    assert_allclose(result.models.parameters["index"].value, 2.7, rtol=0.1)
    assert_allclose(result.models.parameters["amplitude"].value, 2e-11, rtol=0.1)
    assert_allclose(result.models.parameters["index"].error, 0.1, rtol=0.2)
    assert_allclose(result.models.parameters["amplitude"].error, 1.6e-12, rtol=0.2)

    assert result.models._covariance is None

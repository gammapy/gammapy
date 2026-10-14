# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import importlib
import sys
import types
import numpy as np
from gammapy.utils.testing import requires_data
from numpy.testing import assert_allclose
from gammapy.modeling.models import SkyModel
from gammapy.datasets import Datasets, SpectrumDatasetOnOff
from gammapy.modeling.sampler import Sampler, SamplerLikelihood, SAMPLER_BACKENDS
from gammapy.utils.deprecation import GammapyDeprecationWarning
from gammapy.modeling.models import (
    UniformPrior,
    LogUniformPrior,
    PowerLawSpectralModel,
    Models,
)

tested_backends = [_ for _ in SAMPLER_BACKENDS if importlib.util.find_spec(_)]


@pytest.fixture()
def datasets_sampler():
    datasets = Datasets()
    for obs_id in [23523, 23526]:
        dataset = SpectrumDatasetOnOff.read(
            f"$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{obs_id}.fits"
        )
        datasets.append(dataset)
    return datasets


@pytest.fixture()
def models_sampler():
    pwl1 = PowerLawSpectralModel(index=2.3)
    pwl1.amplitude.prior = LogUniformPrior(min=1e-12, max=1e-10)
    pwl1.index.prior = UniformPrior(min=2, max=3)
    return Models([SkyModel(pwl1, name="source1")])


def test_sampler_nautilus_defaults():
    sampler = Sampler(backend="nautilus")

    assert sampler.backend == "nautilus"
    assert sampler.sampler_opts["n_live"] == 2000
    assert sampler.sampler_opts["filepath"] is None
    assert sampler.sampler_opts["resume"] is True
    assert_allclose(sampler.run_opts["f_live"], 0.01)
    assert sampler.run_opts["n_eff"] == 2000


def test_sampler_nautilus_defaults_can_be_overridden():
    sampler = Sampler(
        backend="nautilus",
        sampler_opts={"n_live": 500, "filepath": "test"},
        run_opts={"f_live": 0.001, "n_eff": 500},
    )

    assert sampler.sampler_opts["n_live"] == 500
    assert sampler.sampler_opts["filepath"] == "test"
    assert_allclose(sampler.run_opts["f_live"], 0.001)
    assert sampler.run_opts["n_eff"] == 500


def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="unknown_backend"):
        Sampler(backend="unknown_backend")


def test_sampler_ultranest_defaults():
    sampler = Sampler(backend="ultranest")

    assert sampler.backend == "ultranest"
    assert sampler.sampler_opts["log_dir"] is None
    assert sampler.sampler_opts["resume"] == "subfolder"
    assert sampler.sampler_opts["step_sampler"] is False
    assert sampler.sampler_opts["nsteps"] == 10
    assert sampler.run_opts["min_num_live_points"] == 400
    assert_allclose(sampler.run_opts["frac_remain"], 0.5)


@pytest.mark.parametrize("backend", list(SAMPLER_BACKENDS))
def test_sampler_user_opts_not_mutated(backend):
    sampler_opts = {"resume": "overwrite"}
    run_opts = {"max_ncalls": 10}
    sampler = Sampler(backend=backend, sampler_opts=sampler_opts, run_opts=run_opts)

    assert sampler_opts == {"resume": "overwrite"}
    assert run_opts == {"max_ncalls": 10}
    assert sampler.sampler_opts["resume"] == "overwrite"
    assert sampler.run_opts["max_ncalls"] == 10


@pytest.mark.parametrize(
    ("old_name", "new_name", "value"),
    [("live_points", "min_num_live_points", 123), ("frac_remain", "frac_remain", 0.1)],
)
def test_sampler_ultranest_deprecated_opts(old_name, new_name, value):
    sampler_opts = {old_name: value}
    with pytest.warns(GammapyDeprecationWarning, match=old_name):
        sampler = Sampler(backend="ultranest", sampler_opts=sampler_opts)

    assert sampler_opts == {old_name: value}
    assert old_name not in sampler.sampler_opts
    assert_allclose(sampler.run_opts[new_name], value)


def test_sampler_ultranest_deprecated_opts_run_opts_win():
    with pytest.warns(GammapyDeprecationWarning):
        sampler = Sampler(
            backend="ultranest",
            sampler_opts={"live_points": 100},
            run_opts={"min_num_live_points": 200},
        )
    assert sampler.run_opts["min_num_live_points"] == 200


class _MockBackendSampler:
    """Records the options passed by the Sampler."""

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.run_kwargs = None
        self.stepsampler = None

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        return {"posterior": {"mean": [1.0], "stdev": [0.1]}}


@pytest.fixture()
def mock_ultranest(monkeypatch):
    module = types.ModuleType("ultranest")
    module.ReactiveNestedSampler = _MockBackendSampler
    monkeypatch.setitem(sys.modules, "ultranest", module)


@pytest.fixture()
def sampler_parameters(models_sampler):
    parameters = models_sampler.parameters.free_unique_parameters
    like = SamplerLikelihood(function=lambda: 0.0, parameters=parameters)
    return parameters, like


def test_sampler_ultranest_opts_forwarded(mock_ultranest, sampler_parameters):
    sampler = Sampler(
        backend="ultranest",
        sampler_opts={"vectorized": True, "nsteps": 5},
        run_opts={"max_ncalls": 10},
    )
    sampler._backend.build(*sampler_parameters)
    sampler._backend.run()

    init_kwargs = sampler._sampler.init_kwargs
    assert init_kwargs["vectorized"] is True
    assert init_kwargs["resume"] == "subfolder"
    assert init_kwargs["log_dir"] is None
    assert "step_sampler" not in init_kwargs
    assert "nsteps" not in init_kwargs
    assert sampler._sampler.stepsampler is None

    assert sampler._sampler.run_kwargs == {
        "min_num_live_points": 400,
        "frac_remain": 0.5,
        "max_ncalls": 10,
    }


def test_sampler_nautilus_opts_forwarded(monkeypatch, sampler_parameters):
    class MockNautilusSampler(_MockBackendSampler):
        n_like = 10
        log_z = 0.0

        def run(self, **kwargs):
            self.run_kwargs = kwargs
            return True

        def posterior(self, equal_weight=False):
            points = np.array([[1.0, 2.0], [3.0, 4.0]])
            if equal_weight:
                return (points,)
            return points, np.log([0.5, 0.5]), np.array([0.0, 0.0])

    module = types.ModuleType("nautilus")
    module.Sampler = MockNautilusSampler
    monkeypatch.setitem(sys.modules, "nautilus", module)

    sampler = Sampler(
        backend="nautilus",
        sampler_opts={"n_live": 100, "seed": 42},
        run_opts={"discard_exploration": True},
    )
    sampler._backend.build(*sampler_parameters)
    result = sampler._backend.run()

    init_kwargs = sampler._sampler.init_kwargs
    assert init_kwargs["n_live"] == 100
    assert init_kwargs["seed"] == 42
    assert init_kwargs["resume"] is True
    assert init_kwargs["n_dim"] == 2

    assert sampler._sampler.run_kwargs == {
        "f_live": 0.01,
        "n_eff": 2000,
        "verbose": True,
        "discard_exploration": True,
    }
    assert_allclose(result["posterior"]["mean"], [2.0, 3.0])
    assert sampler._backend.to_result(result).success


@pytest.mark.parametrize("backend", tested_backends)
@requires_data()
def test_run_missing_prior(backend, datasets_sampler, models_sampler):
    models_sampler[0].spectral_model.index.prior = None
    datasets_sampler.models = models_sampler

    sampler = Sampler(backend=backend)
    with pytest.raises(ValueError):
        sampler.run(datasets_sampler)


@pytest.mark.parametrize("backend", tested_backends)
@requires_data()
def test_run(backend, datasets_sampler, models_sampler):
    datasets_sampler.models = models_sampler

    if backend == "ultranest":
        sampler_opts = {}
        run_opts = {"min_num_live_points": 300}
    elif backend == "nautilus":
        sampler_opts = {"n_live": 300}
        run_opts = {"n_eff": 200}
    sampler = Sampler(backend=backend, sampler_opts=sampler_opts, run_opts=run_opts)

    result = sampler.run(datasets_sampler)

    assert result.success
    assert (
        result.samples.shape[1]
        == datasets_sampler.models.parameters.free_unique_parameters.value.shape[0]
    )
    assert (
        result.sampler_results["weighted_samples"]["points"].shape[1]
        == datasets_sampler.models.parameters.free_unique_parameters.value.shape[0]
    )
    assert (
        result.sampler_results["weighted_samples"]["weights"].shape[0]
        == result.sampler_results["weighted_samples"]["points"].shape[0]
    )

    required_keys = [
        "logz",
        "logzerr",
        "posterior",
        "samples",
        "ncall",
        "insertion_order_MWW_test",
    ]

    if backend == "ultranest":
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


@pytest.mark.parametrize("backend", tested_backends)
@requires_data()
def test_run_linked_params(backend, datasets_sampler, models_sampler):
    pwl1 = models_sampler[0].spectral_model
    pwl2 = PowerLawSpectralModel()
    pwl2.index = pwl1.index
    pwl2.amplitude = pwl1.amplitude

    models_sampler.append(SkyModel(pwl2, name="source2"))
    datasets_sampler.models = models_sampler

    if backend == "ultranest":
        sampler_opts = {}
        run_opts = {"min_num_live_points": 300}
    elif backend == "nautilus":
        sampler_opts = {"n_live": 300}
        run_opts = {"n_eff": 200}
    sampler = Sampler(backend=backend, sampler_opts=sampler_opts, run_opts=run_opts)

    result = sampler.run(datasets_sampler)

    assert result.success
    assert (
        result.samples.shape[1]
        == datasets_sampler.models.parameters.free_unique_parameters.value.shape[0]
    )

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

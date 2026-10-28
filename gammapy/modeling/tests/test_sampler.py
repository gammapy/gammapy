# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import sys
import types
import numpy as np
from gammapy.utils.testing import requires_data
from numpy.testing import assert_allclose
from gammapy.modeling import SAMPLER_BACKEND_REGISTRY
from gammapy.modeling.models import SkyModel
from gammapy.datasets import Datasets, SpectrumDatasetOnOff
from gammapy.modeling.sampler import (
    Sampler,
    SamplerLikelihood,
)
from gammapy.utils.deprecation import GammapyDeprecationWarning
from gammapy.modeling.models import (
    UniformPrior,
    LogUniformPrior,
    PowerLawSpectralModel,
    Models,
)


tested_backends = [_.tag for _ in SAMPLER_BACKEND_REGISTRY]


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


@pytest.mark.parametrize(
    ("backend", "defaults"),
    [
        (
            "ultranest",
            {
                "log_dir": None,
                "resume": "subfolder",
                "min_num_live_points": 400,
                "frac_remain": 0.5,
            },
        ),
        (
            "nautilus",
            {
                "n_live": 2000,
                "filepath": None,
                "resume": True,
                "f_live": 0.01,
                "n_eff": 2000,
                "verbose": True,
            },
        ),
    ],
)
def test_sampler_defaults(backend, defaults):
    sampler = Sampler(backend=backend)

    assert sampler.backend == backend
    assert sampler.options == defaults


def test_sampler_options_override_defaults():
    options = {"n_live": 500, "n_eff": 500}
    sampler = Sampler(backend="nautilus", options=options)

    assert options == {"n_live": 500, "n_eff": 500}
    assert sampler.options["n_live"] == 500
    assert sampler.options["n_eff"] == 500
    assert sampler.options["resume"] is True


def test_invalid_backend_raises():
    with pytest.raises(KeyError, match="unknown_backend"):
        Sampler(backend="unknown_backend")


def test_sampler_init_does_not_import_backend(monkeypatch):
    # an import of "ultranest" now raises an ImportError
    monkeypatch.setitem(sys.modules, "ultranest", None)

    sampler = Sampler(backend="ultranest", options={"min_num_live_points": 100})
    assert sampler.options["min_num_live_points"] == 100

    with pytest.raises(ImportError):
        sampler._backend.resolve_options()


class _MockUltranestSampler:
    """Records the options passed to it, with an explicit signature."""

    def __init__(
        self,
        param_names,
        loglike,
        transform=None,
        log_dir=None,
        resume=None,
        vectorized=False,
    ):
        self.init_kwargs = dict(log_dir=log_dir, resume=resume, vectorized=vectorized)

    def run(self, min_num_live_points=400, frac_remain=0.01, max_ncalls=None):
        self.run_kwargs = dict(
            min_num_live_points=min_num_live_points,
            frac_remain=frac_remain,
            max_ncalls=max_ncalls,
        )
        return {"posterior": {"mean": [1.0], "stdev": [0.1]}}


class _MockNautilusSampler:
    """Records the options passed to it, with an explicit signature."""

    n_like = 10
    log_z = 0.0

    def __init__(
        self,
        prior,
        likelihood,
        n_dim,
        n_live=2000,
        filepath=None,
        resume=True,
        seed=None,
    ):
        self.init_kwargs = dict(
            n_dim=n_dim, n_live=n_live, filepath=filepath, resume=resume, seed=seed
        )

    def run(self, f_live=0.01, n_eff=10000, verbose=False, discard_exploration=False):
        self.run_kwargs = dict(
            f_live=f_live,
            n_eff=n_eff,
            verbose=verbose,
            discard_exploration=discard_exploration,
        )
        return True

    def posterior(self, equal_weight=False):
        points = np.array([[1.0, 2.0], [3.0, 4.0]])
        if equal_weight:
            return (points,)
        return points, np.log([0.5, 0.5]), np.array([0.0, 0.0])


@pytest.fixture()
def mock_ultranest(monkeypatch):
    def set_sampler_class(sampler_class=_MockUltranestSampler):
        module = types.ModuleType("ultranest")
        module.ReactiveNestedSampler = sampler_class
        monkeypatch.setitem(sys.modules, "ultranest", module)

    set_sampler_class()
    return set_sampler_class


@pytest.fixture()
def mock_nautilus(monkeypatch):
    module = types.ModuleType("nautilus")
    module.Sampler = _MockNautilusSampler
    monkeypatch.setitem(sys.modules, "nautilus", module)


@pytest.fixture()
def sampler_parameters(models_sampler):
    parameters = models_sampler.parameters.free_unique_parameters
    like = SamplerLikelihood(function=lambda: 0.0, parameters=parameters)
    return parameters, like


def _build_and_run(sampler, sampler_parameters):
    sampler._backend.resolve_options()
    sampler._backend.build(*sampler_parameters)
    return sampler._backend.run()


def test_sampler_ultranest_options_routed(mock_ultranest, sampler_parameters):
    options = {"vectorized": True, "min_num_live_points": 100, "max_ncalls": 10}
    sampler = Sampler(backend="ultranest", options=options)
    _build_and_run(sampler, sampler_parameters)

    assert sampler._sampler.init_kwargs == {
        "log_dir": None,
        "resume": "subfolder",
        "vectorized": True,
    }
    assert sampler._sampler.run_kwargs == {
        "min_num_live_points": 100,
        "frac_remain": 0.5,
        "max_ncalls": 10,
    }


def test_sampler_nautilus_options_routed(mock_nautilus, sampler_parameters):
    options = {"n_live": 100, "seed": 42, "discard_exploration": True}
    sampler = Sampler(backend="nautilus", options=options)
    result = _build_and_run(sampler, sampler_parameters)

    assert sampler._sampler.init_kwargs == {
        "n_dim": 2,
        "n_live": 100,
        "filepath": None,
        "resume": True,
        "seed": 42,
    }
    assert sampler._sampler.run_kwargs == {
        "f_live": 0.01,
        "n_eff": 2000,
        "verbose": True,
        "discard_exploration": True,
    }
    assert_allclose(result["posterior"]["mean"], [2.0, 3.0])
    assert sampler._backend.to_result(result).success


def test_sampler_options_unknown_raises(mock_ultranest):
    # no error when the sampler is created, only when the options are resolved
    sampler = Sampler(backend="ultranest", options={"nlive": 10})
    with pytest.raises(ValueError, match="Unknown ultranest sampler option 'nlive'"):
        sampler._backend.resolve_options()


@pytest.mark.parametrize("name", ["param_names", "loglike", "transform"])
def test_sampler_options_gammapy_filled_args_raises(mock_ultranest, name):
    sampler = Sampler(backend="ultranest", options={name: None})
    with pytest.raises(ValueError, match="Unknown"):
        sampler._backend.resolve_options()


def test_sampler_options_collision_raises(mock_ultranest):
    class CollisionSampler(_MockUltranestSampler):
        def run(self, resume=None):
            pass

    mock_ultranest(CollisionSampler)
    sampler = Sampler(backend="ultranest", options={"resume": True})
    with pytest.raises(ValueError, match="accepted by both"):
        sampler._backend.resolve_options()


def test_sampler_options_var_keyword_raises(mock_ultranest):
    class VarKeywordSampler(_MockUltranestSampler):
        def run(self, **kwargs):
            pass

    mock_ultranest(VarKeywordSampler)
    sampler = Sampler(backend="ultranest", options={"log_dir": None})
    with pytest.raises(ValueError, match="cannot be routed automatically"):
        sampler._backend.resolve_options()


def test_sampler_options_deprecated_name(mock_ultranest):
    with pytest.warns(GammapyDeprecationWarning, match="live_points") as record:
        sampler = Sampler(backend="ultranest", options={"live_points": 123})
    # the warning points to the line of the user
    assert record[0].filename == __file__
    assert sampler.options["min_num_live_points"] == 123
    assert "live_points" not in sampler.options


@pytest.mark.parametrize(
    ("argument", "value"),
    [("sampler_opts", {"vectorized": True}), ("run_opts", {"max_ncalls": 10})],
)
def test_sampler_deprecated_arguments(
    mock_ultranest, sampler_parameters, argument, value
):
    kwargs = {argument: value}
    with pytest.warns(GammapyDeprecationWarning, match=f'"{argument}" was deprecated'):
        sampler = Sampler(backend="ultranest", **kwargs)

    assert kwargs[argument] == value
    _build_and_run(sampler, sampler_parameters)
    if argument == "sampler_opts":
        assert sampler._sampler.init_kwargs["vectorized"] is True
    else:
        assert sampler._sampler.run_kwargs["max_ncalls"] == 10
    # the defaults still apply
    assert sampler._sampler.init_kwargs["resume"] == "subfolder"
    assert sampler._sampler.run_kwargs["min_num_live_points"] == 400


def test_sampler_deprecated_arguments_combined_with_options(mock_ultranest):
    with pytest.warns(GammapyDeprecationWarning):
        sampler = Sampler(
            backend="ultranest",
            options={"log_dir": "out"},
            run_opts={"max_ncalls": 10},
        )
    sampler_opts, run_opts = sampler._backend.resolve_options()
    assert sampler_opts["log_dir"] == "out"
    assert run_opts["max_ncalls"] == 10


def test_sampler_deprecated_arguments_duplicate_raises():
    with pytest.warns(GammapyDeprecationWarning):
        with pytest.raises(ValueError, match="given twice"):
            Sampler(
                backend="ultranest",
                options={"log_dir": "a"},
                sampler_opts={"log_dir": "b"},
            )


@pytest.mark.parametrize(
    ("old_name", "new_name", "value"),
    [("live_points", "min_num_live_points", 123), ("frac_remain", "frac_remain", 0.1)],
)
def test_sampler_ultranest_deprecated_sampler_opts_names(
    mock_ultranest, old_name, new_name, value
):
    sampler_opts = {old_name: value}
    with pytest.warns(GammapyDeprecationWarning) as record:
        sampler = Sampler(backend="ultranest", sampler_opts=sampler_opts)

    messages = [str(warning.message) for warning in record]
    assert all(warning.filename == __file__ for warning in record)
    assert any('"sampler_opts" was deprecated' in message for message in messages)
    assert any(f"'{old_name}' is deprecated" in message for message in messages)

    assert sampler_opts == {old_name: value}
    sampler_opts, run_opts = sampler._backend.resolve_options()
    assert old_name not in sampler_opts
    assert_allclose(run_opts[new_name], value)


def test_sampler_deprecated_properties(mock_ultranest):
    sampler = Sampler(backend="ultranest", options={"max_ncalls": 10})

    with pytest.warns(GammapyDeprecationWarning, match="Sampler.sampler_opts"):
        assert sampler.sampler_opts == {"log_dir": None, "resume": "subfolder"}
    with pytest.warns(GammapyDeprecationWarning, match="Sampler.run_opts"):
        assert sampler.run_opts == {
            "min_num_live_points": 400,
            "frac_remain": 0.5,
            "max_ncalls": 10,
        }


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
        options = {"min_num_live_points": 300}
    elif backend == "nautilus":
        options = {"n_live": 300, "n_eff": 200}
    sampler = Sampler(backend=backend, options=options)

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
        options = {"min_num_live_points": 300}
    elif backend == "nautilus":
        options = {"n_live": 300, "n_eff": 200}
    sampler = Sampler(backend=backend, options=options)

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

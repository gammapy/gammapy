# Licensed under a 3-clause BSD style license - see LICENSE.rst

from numpy.testing import assert_allclose
from gammapy.datasets import Datasets
from gammapy.datasets import SpectrumDatasetOnOff

from gammapy.modeling.models import (
    SkyModel,
    LogUniformPrior,
)
import pytest
from gammapy.modeling.sampler import Sampler
from gammapy.utils.testing import requires_data, requires_dependency
from gammapy.modeling.models import Models
from gammapy.modeling.selection import BayesianModelSelection


@pytest.fixture()
def alternative_models():
    alternative_models = {}
    model_type = "lp"
    prior_types = ["uniformative", "strong"]

    for prior_type in prior_types:
        model_name = f"{model_type}({prior_type})"
        model = SkyModel.create(spectral_model=model_type, name=f"crab-{model_name}")
        model.spectral_model.reference.value = 1
        model.spectral_model.alpha.frozen = True
        model.spectral_model.beta.frozen = True

        if "uniformative" in prior_type:
            model.spectral_model.amplitude.prior = LogUniformPrior(min=1e-14, max=1e-8)

        elif "strong" in prior_type:
            model.spectral_model.amplitude.prior = LogUniformPrior(min=1e-11, max=1e-10)

        alternative_models[model_name] = Models(model)

    return alternative_models


@requires_data()
@requires_dependency("arviz")
def test_bayesian_model_selection(alternative_models):
    path = "$GAMMAPY_DATA/joint-crab/spectra/hess/"

    datasets = Datasets()
    for id in ["23523", "23526", "23559", "23592"]:
        dataset = SpectrumDatasetOnOff.read(f"{path}pha_obs{id}.fits")
        datasets.append(dataset)

    sampler_opts = {
        "live_points": 50,
        "frac_remain": 0.5,
        "log_dir": None,
    }

    sampler = Sampler(backend="ultranest", sampler_opts=sampler_opts)

    bms = BayesianModelSelection(
        sampler, posterior_downsample_factor=5, n_prior_samples=10
    )

    bms_results = bms.run(datasets, alternative_models)

    inference_result = bms_results["lp(uniformative)"]
    assert "Statistics summary in deviance scale" in str(inference_result)

    assert_allclose(inference_result.elpd_loo.p_loo, 1.85, rtol=1e-1)

    assert_allclose(inference_result.priors["amplitude"].entropy(), -22.7, rtol=1e-1)
    bms_results["lp(strong)"].priors["amplitude"].entropy()

    aic = -2 * inference_result.logl + 2 * inference_result.dof
    assert_allclose(inference_result.aic, aic)

    psense = inference_result.prior_sensitivity_table()
    assert_allclose(psense.prior.amplitude, 0.01, rtol=1e-1)
    assert_allclose(psense.likelihood.amplitude, 0.097, rtol=5e-1)

    parameter_table = inference_result.parameters_table()
    assert parameter_table["parameter"] == "amplitude"
    assert_allclose(parameter_table["median"], 5.49e-11, rtol=1e-1)
    assert_allclose(parameter_table["mean"], 5.50e-11, rtol=1e-1)
    assert_allclose(parameter_table["mode"], 5.52e-11, rtol=1e-1)
    assert_allclose(parameter_table["value at max ln(L)"], 5.44e-11, rtol=1e-1)

    stats_table = bms_results.stats_table()
    assert_allclose(stats_table["-2logl"], 216.356, rtol=1e-1)
    assert bms_results["lp(uniformative)"].logz < bms_results["lp(strong)"].logz

    table = bms_results.stats_difference_table("lp(uniformative)")
    test_str = "H0: lp(uniformative) - H1: lp(strong)"
    assert table["Model (prior)"][0] == test_str
    assert_allclose(table["-2logl"], 0, atol=1e-5)

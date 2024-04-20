# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from gammapy.modeling.fit import Fit
from gammapy.modeling.selection import TestStatisticNested, select_nested_models
from gammapy.utils.testing import requires_data


@pytest.fixture(scope="session")
def fermi_datasets():
    from gammapy.datasets import Datasets

    filename = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml"
    filename_models = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml"
    return Datasets.read(filename=filename, filename_models=filename_models)


@requires_data()
def test_test_statistic_detection(fermi_datasets):

    model = fermi_datasets.models["Crab Nebula"]

    results = select_nested_models(
        fermi_datasets, [model.spectral_model.amplitude], [0]
    )
    assert_allclose(results["ts"], 20905.667798, rtol=1e-5)

    ts_eval = TestStatisticNested([model.spectral_model.amplitude], [0])
    ts_known_bkg = ts_eval.ts_known_bkg(fermi_datasets)
    ts_asimov = ts_eval.ts_asimov(fermi_datasets)
    ts = ts_eval.ts(fermi_datasets)
    assert_allclose(ts_known_bkg, 28086.269146, rtol=1e-2)
    assert_allclose(ts_asimov, 27495.908291, rtol=1e-2)
    assert_allclose(ts, 20905.667798, rtol=1e-5)

    # bad model
    bias_factor = 1.2
    model.spectral_model.amplitude.value *= bias_factor
    ts_eval = TestStatisticNested([model.spectral_model.amplitude], [0])
    ts_known_bkg = ts_eval.ts_known_bkg(fermi_datasets)
    ts_asimov = ts_eval.ts_asimov(fermi_datasets)
    ts = ts_eval.ts(fermi_datasets)
    assert_allclose(ts_known_bkg, 28002.894113, rtol=1e-2)
    assert_allclose(ts_asimov, 34130.645399, rtol=1e-2)  # ts_known_bkg<<ts_asimov
    assert_allclose(ts, 20905.667798, rtol=1e-5)
    model.spectral_model.amplitude.value /= bias_factor


@requires_data()
def test_test_statistic_detection_other_frozen(fermi_datasets):

    with fermi_datasets.models.restore_status():
        fermi_datasets.models.freeze()
        model = fermi_datasets.models["Crab Nebula"]
        results = select_nested_models(
            fermi_datasets, [model.spectral_model.amplitude], [0]
        )
        results["fit_results_null"].nfev == 0
        model.spectral_model.amplitude.value = 0
        assert_allclose(
            results["fit_results_null"].parameters.value,
            fermi_datasets.models.parameters.value,
        )


@requires_data()
def test_test_statistic_link(fermi_datasets):

    # TODO: better test with simulated data ?
    model = fermi_datasets.models["Crab Nebula"]
    model2 = model.copy(name="other")
    model2.spectral_model.alpha.value = 2.4
    fermi_datasets.models = fermi_datasets.models + [model2]

    fit = Fit()
    minuit_opts = {"tol": 10, "strategy": 0}
    fit.backend = "minuit"
    fit.optimize_opts = minuit_opts

    ts_eval = TestStatisticNested(
        [model.spectral_model.alpha], [model2.spectral_model.alpha], fit=fit
    )
    results = ts_eval.run(fermi_datasets)

    assert results["ts"] < ts_eval.ts_threshold
    assert_allclose(model2.spectral_model.alpha.value, model.spectral_model.alpha.value)

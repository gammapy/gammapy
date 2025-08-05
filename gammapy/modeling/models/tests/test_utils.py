# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.time import Time
from gammapy.modeling.models import LightCurveTemplateTemporalModel, SkyModel
from gammapy.modeling.models.utils import (
    _template_model_from_cta_sdc,
    read_hermes_cube,
    FluxPredictionBand,
)
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import requires_data, requires_dependency, mpl_plot_check


@requires_data()
def test__template_model_from_cta_sdc(tmp_path):
    filename = "$GAMMAPY_DATA/gravitational_waves/GW_example_DC_file.fits.gz"
    mod = _template_model_from_cta_sdc(filename)

    assert isinstance(mod, LightCurveTemplateTemporalModel)
    t = Time(55555.6157407407, format="mjd")
    val = mod.evaluate(t, energy=[0.3, 2] * u.TeV)
    assert_allclose(val.data, [[2.281216e-21], [4.281390e-23]], rtol=1e-5)

    filename = make_path(tmp_path / "test.fits")
    mod.write(filename=filename, format="map", overwrite=True)
    mod1 = LightCurveTemplateTemporalModel.read(filename=filename, format="map")

    assert_allclose(mod1.t_ref.value, mod.t_ref.value, rtol=1e-7)


@requires_data()
def test__reference_time():
    filename = "$GAMMAPY_DATA/gravitational_waves/GW_example_DC_file.fits.gz"
    t_ref = Time("2028-01-01T00:00:00", format="isot", scale="utc")
    mod = _template_model_from_cta_sdc(filename, t_ref=t_ref)

    assert_allclose(mod.reference_time.mjd, t_ref.mjd, rtol=1e-7)


@requires_data()
@requires_dependency("healpy")
def test_read_hermes_cube():
    filename = make_path(
        "$GAMMAPY_DATA/tests/hermes/hermes-VariableMin-pi0-Htot_CMZ_nside256.fits.gz"
    )
    map_ = read_hermes_cube(filename)
    assert map_.geom.frame == "galactic"
    assert map_.geom.axes[0].unit == "GeV"
    assert_allclose(map_.geom.axes[0].center[3], 1 * u.GeV)
    assert_allclose(map_.get_by_coord((0 * u.deg, 0 * u.deg, 1 * u.GeV)), 2.6391575)


def test_flux_prediction_band_validation():
    model = SkyModel.create(spectral_model="pl")
    samples = {}
    for par in model.parameters:
        samples[par.name] = np.ones(10) * par.quantity

    predict = FluxPredictionBand(model.spectral_model, samples)
    assert predict.model == model.spectral_model
    assert "amplitude" in predict.samples.keys()

    with pytest.raises(TypeError):
        FluxPredictionBand(model, samples)

    bad_samples = samples.copy()
    bad_samples["index"] = np.ones(5)
    with pytest.raises(ValueError):
        FluxPredictionBand(model.spectral_model, bad_samples)

    bad_samples = samples.copy()
    bad_samples.pop("index")
    with pytest.raises(ValueError):
        FluxPredictionBand(model.spectral_model, bad_samples)


def test_prediction_percentiles():
    percentiles = FluxPredictionBand._sigma_to_percentiles(2)
    assert_allclose(percentiles, [2.275013, 97.724987])

    with pytest.raises(ValueError):
        FluxPredictionBand._sigma_to_percentiles(-1)


def test_flux_prediction_band_create_from_covariance():
    model = SkyModel.create(spectral_model="pl").spectral_model
    model.covariance = [
        [0.01, 0, 0],
        [0, 1e-26, 0],
        [0, 0, 0.0],
    ]

    predict = FluxPredictionBand.from_model_covariance(model, n_samples=10000)

    assert predict.samples["index"].shape == (10000,)
    assert_allclose(predict.samples["amplitude"].mean().value, 1e-12, rtol=1e-2)
    assert_allclose(predict.samples["amplitude"].std().value, 1e-13, rtol=1e-2)


# This function tests analytical integration methods
def test_flux_prediction_band_pl():
    from gammapy.modeling.models import PowerLawSpectralModel

    model = PowerLawSpectralModel()
    model.covariance = [
        [0.01, 0, 0],
        [0, 1e-26, 0],
        [0, 0, 0.0],
    ]

    pred = FluxPredictionBand.from_model_covariance(
        model, n_samples=10000, random_state=42
    )

    dnde_errn, dnde_errp = pred.evaluate_error([0.1, 1, 10] * u.TeV)

    assert dnde_errn.shape == (3,)
    assert dnde_errp.shape == (3,)
    assert_allclose(dnde_errn.value, [2.21e-11, 9.91e-14, 2.23e-15], rtol=3e-2)
    assert_allclose(dnde_errp.value, [2.85e-11, 9.91e-14, 2.84e-15], rtol=3e-2)

    dnde_errn, dnde_errp = pred.evaluate_error([1e2, 1e3, 1e4] * u.GeV)
    assert_allclose(
        dnde_errn.to_value("TeV-1cm-2s-1"), [2.21e-11, 9.91e-14, 2.23e-15], rtol=3e-2
    )

    flux_errn, flux_errp = pred.integral_error(
        [0.1, 1, 10] * u.TeV, [1, 10, 100] * u.TeV
    )

    assert flux_errn.shape == (3,)
    assert flux_errp.shape == (3,)
    assert flux_errn.unit == u.Unit("cm-2s-1")
    assert_allclose(flux_errn.value, [1.52e-12, 1.06e-13, 2.50e-14], rtol=3e-2)
    assert_allclose(flux_errp.value, [1.87e-12, 1.18e-13, 3.37e-14], rtol=3e-2)

    eflux_errn, eflux_errp = pred.energy_flux_error(
        [0.1, 1, 10] * u.TeV, [1, 10, 100] * u.TeV
    )

    assert eflux_errn.shape == (3,)
    assert eflux_errp.shape == (3,)
    assert eflux_errn.unit == u.Unit("TeV cm-2s-1")
    assert_allclose(eflux_errn.value, [3.30e-13, 3.22e-13, 6.99e-13], rtol=3e-2)
    assert_allclose(eflux_errp.value, [3.79e-13, 3.81e-13, 9.83e-13], rtol=3e-2)


# This function tests numerical integration methods
def test_flux_prediction_band_lpl():
    from gammapy.modeling.models import LogParabolaSpectralModel

    model = LogParabolaSpectralModel(
        amplitude=3.76e-11 * u.Unit("cm-2 s-1 TeV-1"),
        reference=1 * u.TeV,
        alpha=2.44,
        beta=0.25,
    )
    model.covariance = [
        [1.31e-23, 0, -6.80e-14, 3.04e-13],
        [0, 0, 0, 0],
        [-6.80e-14, 0, 0.00899, 0.00904],
        [3.04e-13, 0, 0.00904, 0.0284],
    ]

    pred = FluxPredictionBand.from_model_covariance(model)

    dnde_errn, dnde_errp = pred.evaluate_error([0.1, 1, 10] * u.TeV)

    assert dnde_errn.shape == (3,)
    assert dnde_errp.shape == (3,)
    assert_allclose(dnde_errn.value, [1.42e-09, 3.55e-12, 2.27e-14], rtol=3e-2)
    assert_allclose(dnde_errp.value, [3.01e-09, 3.62e-12, 6.26e-14], rtol=3e-2)

    flux_errn, flux_errp = pred.integral_error(
        [0.1, 1, 10] * u.TeV, [1, 10, 100] * u.TeV
    )

    assert flux_errn.shape == (3,)
    assert flux_errp.shape == (3,)
    assert flux_errn.unit == u.Unit("cm-2s-1")
    assert_allclose(flux_errn.value, [8.47e-11, 2.56e-12, 9.60e-14], rtol=3e-2)
    assert_allclose(flux_errp.value, [1.38e-10, 3.14e-12, 3.98e-13], rtol=3e-2)

    eflux_errn, eflux_errp = pred.energy_flux_error(
        [0.1, 1, 10] * u.TeV, [1, 10, 100] * u.TeV
    )

    assert eflux_errn.shape == (3,)
    assert eflux_errp.shape == (3,)
    assert eflux_errn.unit == u.Unit("TeV cm-2s-1")
    assert_allclose(eflux_errn.value, [1.33e-11, 6.94e-12, 1.45e-12], rtol=3e-2)
    assert_allclose(eflux_errp.value, [2.05e-11, 1.02e-11, 7.93e-12], rtol=3e-2)

    with mpl_plot_check():
        pred.plot_error([0.1, 10] * u.TeV, sed_type="dnde")

    with mpl_plot_check():
        pred.plot_error([0.1, 10] * u.TeV, sed_type="e2dnde")

    with mpl_plot_check():
        pred.plot_error([0.1, 10] * u.TeV, sed_type="flux")

    with mpl_plot_check():
        pred.plot_error([0.1, 10] * u.TeV, sed_type="eflux")

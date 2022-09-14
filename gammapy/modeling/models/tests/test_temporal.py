# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.table import Table
from astropy.time import Time
from gammapy.data.gti import GTI
from gammapy.modeling.models import (
    TemporalModel,
    ConstantTemporalModel,
    ExpDecayTemporalModel,
    GaussianTemporalModel,
    GeneralizedGaussianTemporalModel,
    LightCurveTemplateTemporalModel,
    LinearTemporalModel,
    PowerLawSpectralModel,
    PowerLawTemporalModel,
    SineTemporalModel,
    SkyModel,
    ConstantSpectralModel,
)
from gammapy.utils.testing import mpl_plot_check, requires_data
from gammapy.modeling.parameter import Parameter
from gammapy.utils.time import time_ref_to_dict


# TODO: add light-curve test case from scratch
# only use the FITS one for I/O (or not at all)
@pytest.fixture(scope="session")
def light_curve():
    path = "$GAMMAPY_DATA/tests/models/light_curve/lightcrv_PKSB1222+216.fits"
    return LightCurveTemplateTemporalModel.read(path)


@requires_data()
def test_light_curve_str(light_curve):
    ss = str(light_curve)
    assert "LightCurveTemplateTemporalModel" in ss


@requires_data()
def test_light_curve_evaluate(light_curve):
    t = Time(59500, format="mjd")
    val = light_curve(t)
    assert_allclose(val, 0.015512, rtol=1e-5)

    t = Time(46300, format="mjd")
    val = light_curve.evaluate(t.mjd, ext=3)
    assert_allclose(val, 0.01551196, rtol=1e-5)


def ph_curve(x, amplitude=0.5, x0=0.01):
    return 100.0 + amplitude * np.sin(2 * np.pi * (x - x0) / 1.0)


def test_time_sampling_template():
    time_ref = Time(55197.00000000, format='mjd')
    livetime = 3.0 * u.hr
    sigma = 0.5 * u.h
    t_min = "2010-01-01T00:00:00"
    t_max = "2010-01-01T03:00:00"
    t_delta = "3 min"

    times = time_ref + livetime * np.linspace(0, 1, 1000)
    flare_model = GaussianTemporalModel(t_ref=(times[500].mjd) * u.d, sigma=sigma)

    lc = Table()
    meta = time_ref_to_dict(times[0])
    lc.meta = meta
    lc.meta["TIMEUNIT"] = 's'
    lc["TIME"] = (times - times[0]).to("s")
    lc["NORM"] = flare_model(times)

    temporal_model = LightCurveTemplateTemporalModel(lc)
    sampler_template = temporal_model.sample_time(
        n_events=1000, t_min=t_min, t_max=t_max, random_state=0, t_delta=t_delta
    )
    assert len(sampler_template) == 1000

    mean = np.mean(sampler_template.mjd)
    std = np.std(sampler_template.mjd)

    assert_allclose(
            mean - times[500].mjd, 0.0,
            atol=1e-3
            )
    assert_allclose(
            std - sigma.to("d").value, 0.0,
            atol=3e-4
            )


def test_time_sampling_gaussian():
    time_ref = Time(55197.00000000, format='mjd')
    sigma = 0.5 * u.h
    t_min = "2010-01-01T00:00:00"
    t_max = "2010-01-01T03:00:00"
    t_delta = "3 min"

    temporal_model = GaussianTemporalModel(t_ref=(time_ref.mjd + 0.03) * u.d, sigma=sigma)
    sampler = temporal_model.sample_time(
        n_events=1000, t_min=t_min, t_max=t_max, random_state=0, t_delta=t_delta
    )
    assert len(sampler) == 1000

    mean = np.mean(sampler.mjd)
    std = np.std(sampler.mjd)
    assert_allclose(
            mean - (time_ref.mjd + 0.03), 0.0,
            atol=4e-3
            )
    assert_allclose(
            std - sigma.to("d").value, 0.0,
            atol=3e-3
            )


def test_lightcurve_temporal_model_integral():
    time = np.arange(0, 10, 0.06) * u.hour
    table = Table()
    table["TIME"] = time
    table["NORM"] = np.ones(len(time))
    table.meta = dict(MJDREFI=55197.0, MJDREFF=0, TIMEUNIT="hour")
    temporal_model = LightCurveTemplateTemporalModel(table)

    start = [1, 3, 5] * u.hour
    stop = [2, 3.5, 6] * u.hour
    gti = GTI.create(start, stop, reference_time=Time("2010-01-01T00:00:00"))

    val = temporal_model.integral(gti.time_start, gti.time_stop)
    assert len(val) == 3
    assert_allclose(np.sum(val), 1.0, rtol=1e-5)


def test_constant_temporal_model_evaluate():
    temporal_model = ConstantTemporalModel()
    t = Time(46300, format="mjd")
    val = temporal_model(t)
    assert_allclose(val, 1.0, rtol=1e-5)


def test_constant_temporal_model_integral():
    temporal_model = ConstantTemporalModel()
    start = [1, 3, 5] * u.day
    stop = [2, 3.5, 6] * u.day
    gti = GTI.create(start, stop)
    val = temporal_model.integral(gti.time_start, gti.time_stop)
    assert len(val) == 3
    assert_allclose(np.sum(val), 1.0, rtol=1e-5)


def test_linear_temporal_model_evaluate():
    t = Time(46301, format="mjd")
    t_ref = 46300 * u.d
    temporal_model = LinearTemporalModel(alpha=1.0, beta=0.1 / u.day, t_ref=t_ref)
    val = temporal_model(t)
    assert_allclose(val, 1.1, rtol=1e-5)


def test_linear_temporal_model_integral():
    t_ref = Time(55555, format="mjd")
    temporal_model = LinearTemporalModel(
        alpha=1.0, beta=0.1 / u.day, t_ref=t_ref.mjd * u.d
    )
    start = [1, 3, 5] * u.day
    stop = [2, 3.5, 6] * u.day
    gti = GTI.create(start, stop, reference_time=t_ref)
    val = temporal_model.integral(gti.time_start, gti.time_stop)
    assert len(val) == 3
    assert_allclose(np.sum(val), 1.345, rtol=1e-5)


def test_exponential_temporal_model_evaluate():
    t = Time(46301, format="mjd")
    t_ref = 46300 * u.d
    t0 = 2.0 * u.d
    temporal_model = ExpDecayTemporalModel(t_ref=t_ref, t0=t0)
    val = temporal_model(t)
    assert_allclose(val, 0.6065306597126334, rtol=1e-5)


def test_exponential_temporal_model_integral():
    t_ref = Time(55555, format="mjd")

    temporal_model = ExpDecayTemporalModel(t_ref=t_ref.mjd * u.d)
    start = [1, 3, 5] * u.day
    stop = [2, 3.5, 6] * u.day
    gti = GTI.create(start, stop, reference_time=t_ref)
    val = temporal_model.integral(gti.time_start, gti.time_stop)
    assert len(val) == 3
    assert_allclose(np.sum(val), 0.102557, rtol=1e-5)


def test_gaussian_temporal_model_evaluate():
    t = Time(46301, format="mjd")
    t_ref = 46300 * u.d
    sigma = 2.0 * u.d
    temporal_model = GaussianTemporalModel(t_ref=t_ref, sigma=sigma)
    val = temporal_model(t)
    assert_allclose(val, 0.882497, rtol=1e-5)


def test_gaussian_temporal_model_integral():
    temporal_model = GaussianTemporalModel(t_ref=50003 * u.d, sigma="2.0 day")
    start = [1, 3, 5] * u.day
    stop = [2, 3.5, 6] * u.day
    t_ref = Time(50000, format="mjd")
    gti = GTI.create(start, stop, reference_time=t_ref)
    val = temporal_model.integral(gti.time_start, gti.time_stop)
    assert len(val) == 3
    assert_allclose(np.sum(val), 0.682679, rtol=1e-5)


def test_generalized_gaussian_temporal_model_evaluate():
    t = Time(46301, format="mjd")
    t_ref = 46300 * u.d
    t_rise = 2.0 * u.d
    t_decay = 2.0 * u.d
    eta = 1 / 2
    temporal_model = GeneralizedGaussianTemporalModel(
        t_ref=t_ref, t_rise=t_rise, t_decay=t_decay, eta=eta
    )
    val = temporal_model(t)
    assert_allclose(val, 0.882497, rtol=1e-5)


def test_generalized_gaussian_temporal_model_integral():
    temporal_model = GeneralizedGaussianTemporalModel(
        t_ref=50003 * u.d, t_rise="2.0 day", t_decay="2.0 day", eta=1 / 2
    )
    start = 1 * u.day
    stop = 2 * u.day
    t_ref = Time(50000, format="mjd")
    gti = GTI.create(start, stop, reference_time=t_ref)
    val = temporal_model.integral(gti.time_start, gti.time_stop)
    assert_allclose(val, 0.759115, rtol=1e-3)


def test_powerlaw_temporal_model_evaluate():
    t = Time(46302, format="mjd")
    t_ref = 46300 * u.d
    alpha = -2.0
    temporal_model = PowerLawTemporalModel(t_ref=t_ref, alpha=alpha)
    val = temporal_model(t)
    assert_allclose(val, 0.25, rtol=1e-5)


def test_powerlaw_temporal_model_integral():
    t_ref = Time(55555, format="mjd")
    temporal_model = PowerLawTemporalModel(alpha=-2.0, t_ref=t_ref.mjd * u.d)
    start = 1 * u.day
    stop = 4 * u.day
    gti = GTI.create(start, stop, reference_time=t_ref)
    val = temporal_model.integral(gti.time_start, gti.time_stop)
    assert len(val) == 1
    assert_allclose(np.sum(val), 0.25, rtol=1e-5)

    temporal_model.parameters["alpha"].value = -1
    start = [1, 3, 5] * u.day
    stop = [2, 3.5, 6] * u.day
    gti = GTI.create(start, stop, reference_time=t_ref)
    val = temporal_model.integral(gti.time_start, gti.time_stop)
    print(np.sum(val))
    assert len(val) == 3
    assert_allclose(np.sum(val), 0.411847, rtol=1e-5)


def test_sine_temporal_model_evaluate():
    t = Time(46302, format="mjd")
    t_ref = 46300 * u.d
    omega = np.pi / 4.0 * u.rad / u.day
    temporal_model = SineTemporalModel(amp=0.5, omega=omega, t_ref=t_ref)
    val = temporal_model(t)
    assert_allclose(val, 1.5, rtol=1e-5)


def test_sine_temporal_model_integral():
    t_ref = Time(55555, format="mjd")
    omega = np.pi / 4.0 * u.rad / u.day
    temporal_model = SineTemporalModel(amp=0.5, omega=omega, t_ref=t_ref.mjd * u.d)
    start = [1, 3, 5] * u.day
    stop = [2, 3.5, 6] * u.day
    gti = GTI.create(start, stop, reference_time=t_ref)
    val = temporal_model.integral(gti.time_start, gti.time_stop)
    assert len(val) == 3
    assert_allclose(np.sum(val), 1.08261, rtol=1e-5)


@requires_data()
def test_to_dict(light_curve):

    out = light_curve.to_dict()
    assert out["temporal"]["type"] == "LightCurveTemplateTemporalModel"
    assert "lightcrv_PKSB1222+216.fits" in out["temporal"]["filename"]


@requires_data()
def test_with_skymodel(light_curve):

    sky_model = SkyModel(spectral_model=PowerLawSpectralModel())
    out = sky_model.to_dict()
    assert "temporal" not in out

    sky_model = SkyModel(
        spectral_model=PowerLawSpectralModel(), temporal_model=light_curve
    )
    assert "LightCurveTemplateTemporalModel" in sky_model.temporal_model.tag

    out = sky_model.to_dict()
    assert "temporal" in out


def test_plot_constant_model():
    time_range = [Time.now(), Time.now() + 1 * u.d]
    constant_model = ConstantTemporalModel(const=1)
    with mpl_plot_check():
        constant_model.plot(time_range)


class MyCustomTemporalModel(TemporalModel):
    """Temporal model with spectral variability

    F(t) = (E/E0)^-\alpha
    where \alpha = (\alpha_0 t/t_ref)^-beta
    """

    tag = "MyCustomTemporalModel"
    is_energy_dependent = True
    beta = Parameter("beta", 0.2)
    _t_ref_default = Time("2000-01-01")
    t_ref = Parameter("t_ref", _t_ref_default.mjd, unit="day", frozen=True)
    E0 = Parameter("E0", "1 TeV", frozen=True)

    @staticmethod
    def evaluate(time, energy, t_ref, beta, E0):
        alpha = np.power((time / t_ref), -beta)
        dim = energy / E0
        if not np.isscalar(dim.value):
            dim = np.expand_dims(dim, axis=1)
        return np.power(dim, -alpha)


@requires_data()
def test_energy_dependent_model():
    t_ref = Time(55555, format="mjd")
    start = [1, 3, 5] * u.day
    stop = [2, 3.5, 6] * u.day
    gti = GTI.create(start, stop, reference_time=t_ref)
    energy = [0.3, 1, 3, 10, 30] * u.TeV

    temporal_model = MyCustomTemporalModel()
    assert temporal_model.is_energy_dependent is True
    val = temporal_model.integral(
        gti.time_start, gti.time_stop, energy=[0.3, 1.0] * u.TeV
    )
    assert val.shape == (2, 3)
    assert_allclose(np.sum(val), 4.3172, rtol=1e-3)

    t = Time(55556, format="mjd")
    val = temporal_model(t, energy=3 * u.TeV)
    assert_allclose(val, 0.3388, rtol=1e-3)

    model = SkyModel(
        spectral_model=ConstantSpectralModel(), temporal_model=temporal_model
    )

    val = model.evaluate(lon=None, lat=None, energy=energy, time=t_ref + start)
    assert_allclose(val.sum().value, 1.425e-11, rtol=1e-3)
    assert val.shape == (5, 3)

    # test evaluation on a dataset
    from gammapy.datasets import MapDataset
    from regions import CircleSkyRegion
    from astropy.coordinates import SkyCoord

    cta_dataset = MapDataset.read(
        "$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz", name="cta_dataset"
    )
    region = CircleSkyRegion(
        center=SkyCoord(0, 0, unit="deg", frame="galactic"), radius=1.0 * u.deg
    )
    ds = cta_dataset.to_spectrum_dataset(region)
    ds.models = model
    assert_allclose(ds.npred().data.sum(), 13172.582827, rtol=1e-3)

    with mpl_plot_check():
        temporal_model.plot(
            time_range=Time([51544.0, 51550], format="mjd"), energy=[0.2, 1] * u.TeV
        )

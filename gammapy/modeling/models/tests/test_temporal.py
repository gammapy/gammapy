# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.table import Table
from astropy.time import Time
from gammapy.modeling.models import (
    ConstantTemporalModel,
    ExpDecayTemporalModel,
    GaussianTemporalModel,
    GeneralizedGaussianTemporalModel,
    LightCurveTemplateTemporalModel,
    LinearTemporalModel,
    Model,
    PowerLawSpectralModel,
    PowerLawTemporalModel,
    SineTemporalModel,
    SkyModel,
    TemplatePhaseCurveTemporalModel,
)
from gammapy.utils.scripts import make_path
from gammapy.utils.testing import mpl_plot_check, requires_data
from gammapy.utils.time import time_ref_to_dict


# TODO: add light-curve test case from scratch
# only use the FITS one for I/O (or not at all)
@pytest.fixture(scope="session")
def light_curve():
    path = "$GAMMAPY_DATA/tests/models/light_curve/lightcrv_PKSB1222+216.fits"
    return LightCurveTemplateTemporalModel.read(path)


@pytest.fixture()
def phase_curve_table():
    phase = np.linspace(0.0, 1, 101)
    norm = phase * (phase < 0.5) + (1 - phase) * (phase >= 0.5)
    return Table(data={"PHASE": phase, "NORM": norm})


@requires_data()
def test_light_curve_str(light_curve):
    ss = str(light_curve)
    assert "LightCurveTemplateTemporalModel" in ss


@requires_data()
def test_light_curve_evaluate(light_curve):
    t = Time(59500, format="mjd")
    val = light_curve(t)
    assert_allclose(val, 0.015512, rtol=1e-5)


@requires_data()
def test_energy_dependent_lightcurve(tmp_path):
    filename = "$GAMMAPY_DATA/gravitational_waves/GW_example_DC_map_file.fits.gz"
    mod = LightCurveTemplateTemporalModel.read(filename, format="map")

    assert mod.is_energy_dependent is True

    t = Time(55555.6157407407, format="mjd")
    val = mod.evaluate(t, energy=[0.3, 2] * u.TeV)
    assert_allclose(val.data, [[2.278068e-21], [4.280503e-23]], rtol=1e-5)

    t = Time([55555, 55556, 55557], format="mjd")
    val = mod.evaluate(t)
    assert val.data.shape == (41, 3)

    with mpl_plot_check():
        mod.plot(
            time_range=(Time(55555.50, format="mjd"), Time(55563.0, format="mjd")),
            energy=[0.3, 2, 10.0] * u.TeV,
        )
    filename = make_path(tmp_path / "test.fits")
    with pytest.raises(NotImplementedError):
        mod.write(filename=filename, format="table", overwrite=True)
    with pytest.raises(NotImplementedError):
        time_start = Time("2010-01-01T00:00:00") + [1, 3, 5] * u.hour
        time_stop = Time("2010-01-01T00:00:00") + [2, 3.5, 6] * u.hour
        mod.integral(time_start, time_stop)


def ph_curve(x, amplitude=0.5, x0=0.01):
    return 100.0 + amplitude * np.sin(2 * np.pi * (x - x0) / 1.0)


@requires_data()
def test_light_curve_to_from_table(light_curve):
    table = light_curve.to_table()
    assert_allclose(table.meta["MJDREFI"], 59000)
    assert_allclose(table.meta["MJDREFF"], 0.4991992, rtol=1e-6)
    assert table.meta["TIMESYS"] == "utc"
    lc1 = LightCurveTemplateTemporalModel.from_table(table)
    assert lc1.map == light_curve.map
    assert_allclose(
        lc1.reference_time.value, Time(59000.5, format="mjd").value, rtol=1e-2
    )

    # test failing cases
    table1 = table.copy()
    table1["TIME"].unit = None
    table.meta = None
    with pytest.raises(ValueError, match="Time unit not found in the table"):
        LightCurveTemplateTemporalModel.from_table(table1)


@requires_data()
def test_light_curve_to_dict(light_curve):
    data = light_curve.to_dict()
    assert data["temporal"]["format"] == "table"
    assert data["temporal"]["unit"] == ""
    assert data["temporal"]["type"] == "LightCurveTemplateTemporalModel"
    assert data["temporal"]["parameters"][0]["name"] == "t_ref"

    lc1 = LightCurveTemplateTemporalModel.from_dict(data)
    assert lc1.map == light_curve.map
    assert_allclose(
        lc1.reference_time.value, light_curve.reference_time.value, rtol=1e-9
    )


@requires_data()
def test_light_curve_map_serialisation(light_curve, tmp_path):
    filename = str(make_path(tmp_path / "tmp.fits"))
    light_curve.write(filename, format="map")
    lc1 = LightCurveTemplateTemporalModel.read(filename, format="map")
    assert_allclose(
        lc1.reference_time.value, light_curve.reference_time.value, rtol=1e-9
    )
    assert lc1.map == light_curve.map


def test_time_sampling_template():
    time_ref = Time(55197.00000000, format="mjd")
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
    lc.meta["TIMEUNIT"] = "s"
    lc["TIME"] = (times - times[0]).to("s")
    lc["NORM"] = flare_model(times)

    temporal_model = LightCurveTemplateTemporalModel.from_table(lc)
    sampler_template = temporal_model.sample_time(
        n_events=1000, t_min=t_min, t_max=t_max, random_state=0, t_delta=t_delta
    )
    assert len(sampler_template) == 1000

    mean = np.mean(sampler_template.mjd)
    std = np.std(sampler_template.mjd)

    assert_allclose(mean - times[500].mjd, 0.0, atol=1e-3)
    assert_allclose(std - sigma.to("d").value, 0.0, atol=3e-4)


def test_time_sampling_gaussian():
    time_ref = Time(55197.00000000, format="mjd")
    sigma = 0.5 * u.h
    t_min = "2010-01-01T00:00:00"
    t_max = "2010-01-01T03:00:00"
    t_delta = "3 min"

    temporal_model = GaussianTemporalModel(
        t_ref=(time_ref.mjd + 0.03) * u.d, sigma=sigma
    )
    sampler = temporal_model.sample_time(
        n_events=1000, t_min=t_min, t_max=t_max, random_state=0, t_delta=t_delta
    )
    assert len(sampler) == 1000

    mean = np.mean(sampler.mjd)
    std = np.std(sampler.mjd)
    assert_allclose(mean - (time_ref.mjd + 0.03), 0.0, atol=4e-3)
    assert_allclose(std - sigma.to("d").value, 0.0, atol=3e-3)


def test_lightcurve_temporal_model_integral():
    time = np.arange(0, 10, 0.06) * u.hour
    table = Table()
    table["TIME"] = time
    table["NORM"] = np.ones(len(time))
    table.meta = dict(MJDREFI=55197.0, MJDREFF=0, TIMEUNIT="hour")
    temporal_model = LightCurveTemplateTemporalModel.from_table(table)
    assert not temporal_model.is_energy_dependent

    time_start = Time("2010-01-01T00:00:00") + [1, 3, 5] * u.hour
    time_stop = Time("2010-01-01T00:00:00") + [2, 3.5, 6] * u.hour
    val = temporal_model.integral(time_start, time_stop)
    assert len(val) == 3
    assert_allclose(np.sum(val), 1.0101, rtol=1e-5)

    with mpl_plot_check():
        temporal_model.plot(
            time_range=(Time(55555.50, format="mjd"), Time(55563.0, format="mjd"))
        )


def test_constant_temporal_model_evaluate():
    temporal_model = ConstantTemporalModel()
    t = Time(46300, format="mjd")
    val = temporal_model(t)
    assert_allclose(val, 1.0, rtol=1e-5)


def test_constant_temporal_model_integral():
    temporal_model = ConstantTemporalModel()
    time_start = Time("2010-01-01T00:00:00") + [1, 3, 5] * u.day
    time_stop = Time("2010-01-01T00:00:00") + [2, 3.5, 6] * u.day
    val = temporal_model.integral(time_start, time_stop)
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
    time_start = t_ref + [1, 3, 5] * u.day
    time_stop = t_ref + [2, 3.5, 6] * u.day
    val = temporal_model.integral(time_start, time_stop)
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
    time_start = t_ref + [1, 3, 5] * u.day
    time_stop = t_ref + [2, 3.5, 6] * u.day
    val = temporal_model.integral(time_start, time_stop)
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
    t_ref = Time(50000, format="mjd")
    time_start = t_ref + [1, 3, 5] * u.day
    time_stop = t_ref + [2, 3.5, 6] * u.day
    val = temporal_model.integral(time_start, time_stop)
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
    t_ref = Time(50000, format="mjd", scale="utc")
    time_start = t_ref + start
    time_stop = t_ref + stop
    val = temporal_model.integral(time_start, time_stop)
    assert_allclose(val, 0.758918, rtol=1e-4)


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
    time_start = t_ref + [1] * u.day
    time_stop = t_ref + [4] * u.day
    val = temporal_model.integral(time_start, time_stop)
    assert len(val) == 1
    assert_allclose(np.sum(val), 0.25, rtol=1e-5)

    temporal_model.parameters["alpha"].value = -1
    time_start = t_ref + [1, 3, 5] * u.day
    time_stop = t_ref + [2, 3.5, 6] * u.day
    val = temporal_model.integral(time_start, time_stop)

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
    time_start = t_ref + [1, 3, 5] * u.day
    time_stop = t_ref + [2, 3.5, 6] * u.day
    val = temporal_model.integral(time_start, time_stop)
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


def test_phase_curve_model(tmp_path):
    phase = np.linspace(0.0, 1, 101)
    norm = phase * (phase < 0.5) + (1 - phase) * (phase >= 0.5)
    table = Table(data={"PHASE": phase, "NORM": norm})

    t_ref = Time("2022-06-01")
    phase_model = TemplatePhaseCurveTemporalModel(
        table=table,
        f0="20 Hz",
        phi_ref=0.0,
        f1="0 s-2",
        f2="0 s-3",
        t_ref=t_ref.mjd * u.d,
    )

    result = phase_model(t_ref + [0, 0.025, 0.05] * u.s)
    assert_allclose(result, [0, 0.5, 0], atol=1e-5)

    phase_model.filename = str(make_path(tmp_path / "tmp.fits"))
    phase_model.write()

    model_dict = phase_model.to_dict()
    new_model = Model.from_dict(model_dict)

    assert_allclose(phase_model.parameters.value, new_model.parameters.value)
    assert phase_model.parameters.names == new_model.parameters.names
    assert (
        phase_model.parameters.free_parameters.names
        == new_model.parameters.free_parameters.names
    )

    assert_allclose(new_model.table["PHASE"].data, phase)
    assert_allclose(new_model.table["NORM"].data, norm)

    # exact number of phases
    integral = phase_model.integral(t_ref, t_ref + 10 * u.s)
    assert_allclose(integral, 0.25, rtol=1e-5)
    # long duration. Should be equal to the phase average
    integral = phase_model.integral(t_ref + 1 * u.h, t_ref + 3 * u.h)
    assert_allclose(integral, 0.25, rtol=1e-5)
    # 1.25 phase
    integral = phase_model.integral(t_ref, t_ref + 62.5 * u.ms)
    assert_allclose(integral, 0.225, rtol=1e-5)


def test_phase_curve_model_sample_time():
    phase = np.linspace(0.0, 1, 51)
    norm = 1 * (phase < 0.5)
    table = Table(data={"PHASE": phase, "NORM": norm})

    t_ref = Time("2020-06-01", scale="utc")
    phase_model = TemplatePhaseCurveTemporalModel(
        table=table,
        f0="50 Hz",
        phi_ref=0.0,
        f1="0 s-2",
        f2="0 s-3",
        t_ref=t_ref.mjd * u.d,
        scale="utc",
    )

    tmin = Time("2023-06-01", scale="tt")
    tmax = tmin + 0.5 * u.h

    times = phase_model.sample_time(10, tmin, tmax)
    phases, _ = phase_model._time_to_phase(
        times,
        phase_model.reference_time,
        phase_model.phi_ref.quantity,
        phase_model.f0.quantity,
        phase_model.f1.quantity,
        phase_model.f2.quantity,
    )

    assert np.all(phases <= 0.5)


@requires_data()
def test_phasecurve_DC1():
    filename = "$GAMMAPY_DATA/tests/phasecurve_LSI_DC.fits"
    t_ref = 43366.275 * u.d
    P0 = 26.7 * u.d
    f0 = 1 / P0

    model = TemplatePhaseCurveTemporalModel.read(filename, t_ref, 0.0, f0)

    times = Time(t_ref, format="mjd") + [0.0, 0.5, 0.65, 1.0] * P0
    norm = model(times)

    assert_allclose(norm, [0.05, 0.15, 1.0, 0.05])

    with mpl_plot_check():
        model.plot_phasogram(n_points=200)


def test_model_scale():
    model = GaussianTemporalModel(t_ref=50003.2503033 * u.d, sigma="2.43 day")
    assert model.scale == "utc"
    model.scale = "tai"
    assert_allclose(model.reference_time.mjd, 50003.2503033, rtol=1e-9)
    dict1 = model.to_dict()
    model1 = GaussianTemporalModel.from_dict(dict1)
    assert model1.scale == "tai"
    assert_allclose(model1.sigma.quantity, 2.43 * u.d, rtol=1e-3)
    time_start = model.reference_time + [1, 3, 5] * u.day
    time_stop = model.reference_time + [2, 3.5, 6] * u.day
    val = model.integral(time_start, time_stop)
    assert_allclose(np.sum(val), 0.442833, rtol=1e-5)

    model1.reference_time = Time(52398.23456, format="mjd", scale="utc")
    assert model1.scale == "tai"
    assert_allclose(model1.t_ref.value, 52398.23493, rtol=1e-9)

    with pytest.raises(TypeError):
        model1.reference_time = 23456

    with pytest.raises(ValueError):
        model = GaussianTemporalModel(
            t_ref=50003.2503033 * u.d, sigma="2.43 day", scale="ms"
        )


@requires_data()
def test_template_temporal_model_format():
    temporal_model = LightCurveTemplateTemporalModel.read(
        "$GAMMAPY_DATA/gravitational_waves/GW_example_DC_map_file.fits.gz", format="map"
    )
    mod_dict = temporal_model.to_dict()
    assert mod_dict["temporal"]["format"] == "map"

    path = "$GAMMAPY_DATA/tests/models/light_curve/lightcrv_PKSB1222+216.fits"
    temporal_model = LightCurveTemplateTemporalModel.read(path)
    mod_dict = temporal_model.to_dict()
    assert mod_dict["temporal"]["format"] == "table"

# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
import astropy.units as u
from gammapy.modeling.models import (
    SPECTRAL_MODELS,
    AbsorbedSpectralModel,
    Absorption,
    ConstantSpectralModel,
    ExpCutoffPowerLaw3FGLSpectralModel,
    ExpCutoffPowerLawSpectralModel,
    GaussianSpectralModel,
    LogGaussianSpectralModel,
    LogParabolaSpectralModel,
    NaimaSpectralModel,
    PowerLaw2SpectralModel,
    PowerLawSpectralModel,
    SuperExpCutoffPowerLaw4FGLSpectralModel,
    TemplateSpectralModel,
)
from gammapy.utils.energy import energy_logspace
from gammapy.utils.testing import (
    assert_quantity_allclose,
    mpl_plot_check,
    requires_data,
    requires_dependency,
)


def table_model():
    energy_edges = energy_logspace(0.1 * u.TeV, 100 * u.TeV, 1000)
    energy = np.sqrt(energy_edges[:-1] * energy_edges[1:])

    model = PowerLawSpectralModel(
        index=2.3, amplitude="4 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    dnde = model(energy)

    return TemplateSpectralModel(energy, dnde, 1)


TEST_MODELS = [
    dict(
        name="powerlaw",
        model=PowerLawSpectralModel(
            index=2.3 * u.Unit(""),
            amplitude=4 / u.cm ** 2 / u.s / u.TeV,
            reference=1 * u.TeV,
        ),
        val_at_2TeV=u.Quantity(4 * 2.0 ** (-2.3), "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(2.9227116204223784, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(6.650836884969039, "TeV cm-2 s-1"),
    ),
    dict(
        name="powerlaw",
        model=PowerLawSpectralModel(
            index=2 * u.Unit(""),
            amplitude=4 / u.cm ** 2 / u.s / u.TeV,
            reference=1 * u.TeV,
        ),
        val_at_2TeV=u.Quantity(1.0, "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(3.6, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(9.210340371976184, "TeV cm-2 s-1"),
    ),
    dict(
        name="powerlaw2",
        model=PowerLaw2SpectralModel(
            amplitude=u.Quantity(2.9227116204223784, "cm-2 s-1"),
            index=2.3 * u.Unit(""),
            emin=1 * u.TeV,
            emax=10 * u.TeV,
        ),
        val_at_2TeV=u.Quantity(4 * 2.0 ** (-2.3), "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(2.9227116204223784, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(6.650836884969039, "TeV cm-2 s-1"),
    ),
    dict(
        name="ecpl",
        model=ExpCutoffPowerLawSpectralModel(
            index=1.6 * u.Unit(""),
            amplitude=4 / u.cm ** 2 / u.s / u.TeV,
            reference=1 * u.TeV,
            lambda_=0.1 / u.TeV,
        ),
        val_at_2TeV=u.Quantity(1.080321705479446, "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(3.765838739678921, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(9.901735870666526, "TeV cm-2 s-1"),
        e_peak=4 * u.TeV,
    ),
    dict(
        name="ecpl_3fgl",
        model=ExpCutoffPowerLaw3FGLSpectralModel(
            index=2.3 * u.Unit(""),
            amplitude=4 / u.cm ** 2 / u.s / u.TeV,
            reference=1 * u.TeV,
            ecut=10 * u.TeV,
        ),
        val_at_2TeV=u.Quantity(0.7349563611124971, "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(2.6034046173089, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(5.340285560055799, "TeV cm-2 s-1"),
    ),
    dict(
        name="plsec_4fgl",
        model=SuperExpCutoffPowerLaw4FGLSpectralModel(
            index_1=1.5,
            index_2=2,
            amplitude=1 / u.cm ** 2 / u.s / u.TeV,
            reference=1 * u.TeV,
            expfactor=1e-2,
        ),
        val_at_2TeV=u.Quantity(0.3431043087721737, "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(1.2125247, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(3.38072082, "TeV cm-2 s-1"),
    ),
    dict(
        name="logpar",
        model=LogParabolaSpectralModel(
            alpha=2.3 * u.Unit(""),
            amplitude=4 / u.cm ** 2 / u.s / u.TeV,
            reference=1 * u.TeV,
            beta=0.5 * u.Unit(""),
        ),
        val_at_2TeV=u.Quantity(0.6387956571420305, "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(2.255689748270628, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(3.9586515834989267, "TeV cm-2 s-1"),
        e_peak=0.74082 * u.TeV,
    ),
    dict(
        name="logpar10",
        model=LogParabolaSpectralModel.from_log10(
            alpha=2.3 * u.Unit(""),
            amplitude=4 / u.cm ** 2 / u.s / u.TeV,
            reference=1 * u.TeV,
            beta=1.151292546497023 * u.Unit(""),
        ),
        val_at_2TeV=u.Quantity(0.6387956571420305, "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(2.255689748270628, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(3.9586515834989267, "TeV cm-2 s-1"),
        e_peak=0.74082 * u.TeV,
    ),
    dict(
        name="constant",
        model=ConstantSpectralModel(const=4 / u.cm ** 2 / u.s / u.TeV),
        val_at_2TeV=u.Quantity(4, "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(35.9999999999999, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(198.00000000000006, "TeV cm-2 s-1"),
    ),
    dict(
        name="powerlaw_index1",
        model=PowerLawSpectralModel(
            index=1 * u.Unit(""),
            amplitude=2 / u.cm ** 2 / u.s / u.TeV,
            reference=1 * u.TeV,
        ),
        val_at_2TeV=u.Quantity(1.0, "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(4.605170185, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(18.0, "TeV cm-2 s-1"),
    ),
    dict(
        name="ecpl_2",
        model=ExpCutoffPowerLawSpectralModel(
            index=2.0 * u.Unit(""),
            amplitude=4 / u.cm ** 2 / u.s / u.TeV,
            reference=1 * u.TeV,
            lambda_=0.1 / u.TeV,
        ),
        val_at_2TeV=u.Quantity(0.81873075, "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(2.83075297, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(6.41406327, "TeV cm-2 s-1"),
        e_peak=np.nan * u.TeV,
    ),
    dict(
        name="GaussianSpectralModel",
        model=GaussianSpectralModel(
            norm=4 / u.cm ** 2 / u.s, mean=2 * u.TeV, sigma=0.2 * u.TeV
        ),
        val_at_2TeV=u.Quantity(7.978845608028654, "cm-2 s-1 TeV-1"),
        val_at_3TeV=u.Quantity(2.973439029468601e-05, "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(3.9999988533937123, "cm-2 s-1"),
        integral_infinity=u.Quantity(4, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(7.999998896163037, "TeV cm-2 s-1"),
    ),
    dict(
        name="LogGaussianSpectralModel",
        model=LogGaussianSpectralModel(
            norm=4 / u.cm ** 2 / u.s, mean=2 * u.TeV, sigma=0.2
        ),
        val_at_2TeV=u.Quantity(3.98942280401, "cm-2 s-1 TeV-1"),
        val_at_3TeV=u.Quantity(0.34066933236079916, "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(3.994439, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(8.151414, "TeV cm-2 s-1"),
    ),
]

# Add compound models

TEST_MODELS.append(
    dict(
        name="compound3",
        model=TEST_MODELS[0]["model"] + TEST_MODELS[0]["model"],
        val_at_2TeV=TEST_MODELS[0]["val_at_2TeV"] * 2,
        integral_1_10TeV=TEST_MODELS[0]["integral_1_10TeV"] * 2,
        eflux_1_10TeV=TEST_MODELS[0]["eflux_1_10TeV"] * 2,
    )
)

TEST_MODELS.append(
    dict(
        name="compound4",
        model=TEST_MODELS[0]["model"] - 0.1 * TEST_MODELS[0]["val_at_2TeV"],
        val_at_2TeV=0.9 * TEST_MODELS[0]["val_at_2TeV"],
        integral_1_10TeV=2.1919819216346936 * u.Unit("cm-2 s-1"),
        eflux_1_10TeV=2.6322140512045697 * u.Unit("TeV cm-2 s-1"),
    )
)

TEST_MODELS.append(
    dict(
        name="compound6",
        model=TEST_MODELS[8]["model"] + u.Quantity(4, "cm-2 s-1 TeV-1"),
        val_at_2TeV=TEST_MODELS[8]["val_at_2TeV"] * 2,
        integral_1_10TeV=TEST_MODELS[8]["integral_1_10TeV"] * 2,
        eflux_1_10TeV=TEST_MODELS[8]["eflux_1_10TeV"] * 2,
    )
)

TEST_MODELS.append(
    dict(
        name="table_model",
        model=table_model(),
        # Values took from power law expectation
        val_at_2TeV=u.Quantity(4 * 2.0 ** (-2.3), "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(2.9227116204223784, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(6.650836884969039, "TeV cm-2 s-1"),
    )
)


@requires_dependency("uncertainties")
@requires_dependency("scipy")
@pytest.mark.parametrize("spectrum", TEST_MODELS, ids=lambda _: _["name"])
def test_models(spectrum):
    model = spectrum["model"]
    energy = 2 * u.TeV
    value = model(energy)
    assert_quantity_allclose(value, spectrum["val_at_2TeV"])
    if "val_at_3TeV" in spectrum:
        energy = 3 * u.TeV
        value = model(energy)
        assert_quantity_allclose(value, spectrum["val_at_3TeV"])

    emin = 1 * u.TeV
    emax = 10 * u.TeV
    assert_quantity_allclose(
        model.integral(emin=emin, emax=emax), spectrum["integral_1_10TeV"]
    )
    assert_quantity_allclose(
        model.energy_flux(emin=emin, emax=emax), spectrum["eflux_1_10TeV"]
    )

    if "e_peak" in spectrum:
        assert_quantity_allclose(model.e_peak, spectrum["e_peak"], rtol=1e-2)

    # inverse for ConstantSpectralModel is irrelevant.
    # inverse for Gaussian has a degeneracy
    if not (
        isinstance(model, ConstantSpectralModel)
        or spectrum["name"] == "compound6"
        or spectrum["name"] == "GaussianSpectralModel"
        or spectrum["name"] == "LogGaussianSpectralModel"
    ):
        assert_quantity_allclose(model.inverse(value), 2 * u.TeV, rtol=0.01)

    if "integral_infinity" in spectrum:
        emin = 0 * u.TeV
        emax = 10000 * u.TeV
        assert_quantity_allclose(
            model.integral(emin=emin, emax=emax), spectrum["integral_infinity"]
        )

    model.to_dict()

    assert "" in str(model)

    # check that an array evaluation works (otherwise e.g. plotting raises an error)
    e_array = [2, 10, 20] * u.TeV
    e_array = e_array[:, np.newaxis, np.newaxis]
    val = model(e_array)
    assert val.shape == e_array.shape
    assert_quantity_allclose(val[0], spectrum["val_at_2TeV"])


def test_model_unit():
    pwl = PowerLawSpectralModel()
    value = pwl(500 * u.MeV)
    assert value.unit == "cm-2 s-1 TeV-1"


@requires_dependency("matplotlib")
@requires_dependency("uncertainties")
def test_model_plot():
    pwl = PowerLawSpectralModel(
        amplitude=1e-12 * u.Unit("TeV-1 cm-2 s-1"), reference=1 * u.Unit("TeV"), index=2
    )
    pwl.parameters.set_parameter_errors(
        {"amplitude": 0.1e-12 * u.Unit("TeV-1 cm-2 s-1")}
    )
    with mpl_plot_check():
        pwl.plot((1 * u.TeV, 10 * u.TeV))

    with mpl_plot_check():
        pwl.plot_error((1 * u.TeV, 10 * u.TeV))


def test_to_from_dict():
    spectrum = TEST_MODELS[0]
    model = spectrum["model"]

    model_dict = model.to_dict()
    model_class = SPECTRAL_MODELS.get_cls(model_dict["type"])
    new_model = model_class.from_dict(model_dict)

    assert isinstance(new_model, PowerLawSpectralModel)

    actual = [par.value for par in new_model.parameters]
    desired = [par.value for par in model.parameters]
    assert_quantity_allclose(actual, desired)


@requires_dependency("matplotlib")
@requires_data()
def test_table_model_from_file():
    filename = "$GAMMAPY_DATA/ebl/ebl_franceschini.fits.gz"
    absorption_z03 = TemplateSpectralModel.read_xspec_model(
        filename=filename, param=0.3
    )
    with mpl_plot_check():
        absorption_z03.plot(energy_range=(0.03, 10), energy_unit=u.TeV, flux_unit="")


@requires_data()
def test_absorption():
    # absorption values for given redshift
    redshift = 0.117
    absorption = Absorption.read_builtin("dominguez")

    # Spectral model corresponding to PKS 2155-304 (quiescent state)
    index = 3.53
    amplitude = 1.81 * 1e-12 * u.Unit("cm-2 s-1 TeV-1")
    reference = 1 * u.TeV
    pwl = PowerLawSpectralModel(index=index, amplitude=amplitude, reference=reference)

    # EBL + PWL model
    model = AbsorbedSpectralModel(
        spectral_model=pwl, absorption=absorption, parameter=redshift
    )

    desired = u.Quantity(5.140765e-13, "TeV-1 s-1 cm-2")
    assert_quantity_allclose(model(1 * u.TeV), desired, rtol=1e-3)
    assert model.alpha_norm.value == 1.0

    # EBL + PWL model: test if norm of EBL=0: it mean model =pwl
    model = AbsorbedSpectralModel(
        spectral_model=pwl, absorption=absorption, alpha_norm=0, parameter=redshift
    )
    assert_quantity_allclose(model(1 * u.TeV), pwl(1 * u.TeV), rtol=1e-3)

    # EBL + PWL model: Test with a norm different of 1
    model = AbsorbedSpectralModel(
        spectral_model=pwl, absorption=absorption, alpha_norm=1.5, parameter=redshift
    )
    desired = u.Quantity(2.739695e-13, "TeV-1 s-1 cm-2")
    assert model.alpha_norm.value == 1.5
    assert_quantity_allclose(model(1 * u.TeV), desired, rtol=1e-3)


@requires_dependency("uncertainties")
def test_pwl_index_2_error():
    pwl = PowerLawSpectralModel(
        amplitude=1e-12 * u.Unit("TeV-1 cm-2 s-1"), reference=1 * u.Unit("TeV"), index=2
    )
    pwl.parameters.set_parameter_errors(
        {"amplitude": 0.1e-12 * u.Unit("TeV-1 cm-2 s-1")}
    )

    val, val_err = pwl.evaluate_error(1 * u.TeV)
    assert_quantity_allclose(val, 1e-12 * u.Unit("TeV-1 cm-2 s-1"))
    assert_quantity_allclose(val_err, 0.1e-12 * u.Unit("TeV-1 cm-2 s-1"))

    flux, flux_err = pwl.integral_error(1 * u.TeV, 10 * u.TeV)
    assert_quantity_allclose(flux, 9e-13 * u.Unit("cm-2 s-1"))
    assert_quantity_allclose(flux_err, 9e-14 * u.Unit("cm-2 s-1"))

    eflux, eflux_err = pwl.energy_flux_error(1 * u.TeV, 10 * u.TeV)
    assert_quantity_allclose(eflux, 2.302585e-12 * u.Unit("TeV cm-2 s-1"))
    assert_quantity_allclose(eflux_err, 0.2302585e-12 * u.Unit("TeV cm-2 s-1"))


def test_ecpl_integrate():
    # regression test to check the numerical integration for small energy bins
    ecpl = ExpCutoffPowerLawSpectralModel()
    value = ecpl.integral(1 * u.TeV, 1.1 * u.TeV)
    assert_quantity_allclose(value, 8.380714e-14 * u.Unit("s-1 cm-2"))


def test_pwl_pivot_energy():
    pwl = PowerLawSpectralModel(amplitude="5.35510540e-11 cm-2 s-1 TeV-1")

    pwl.parameters.covariance = [
        [0.0318377 ** 2, 6.56889442e-14, 0],
        [6.56889442e-14, 0, 0],
        [0, 0, 0],
    ]

    assert_quantity_allclose(pwl.pivot_energy, 3.3540034240210987 * u.TeV)


def test_TemplateSpectralModel_evaluate_tiny():
    energy = np.array([1.00000000e06, 1.25892541e06, 1.58489319e06, 1.99526231e06])
    values = np.array([4.39150790e-38, 1.96639562e-38, 8.80497507e-39, 3.94262401e-39])

    model = TemplateSpectralModel(
        energy=energy, values=values * u.Unit("MeV-1 s-1 sr-1")
    )
    result = model.evaluate(energy, norm=1.0, tilt=0.0, reference=1 * u.TeV)
    tiny = np.finfo(np.float32).tiny
    mask = abs(values) - tiny > tiny
    np.testing.assert_allclose(
        values[mask] / values.max(), result[mask].value / values.max()
    )
    mask = abs(result.value) - tiny <= tiny
    assert np.all(result[mask] == 0.0)


@requires_dependency("naima")
class TestNaimaModel:
    # Used to test model value at 2 TeV
    energy = 2 * u.TeV

    # Used to test model integral and energy flux
    emin = 1 * u.TeV
    emax = 10 * u.TeV

    # Used to that if array evaluation works
    e_array = [2, 10, 20] * u.TeV
    e_array = e_array[:, np.newaxis, np.newaxis]

    def test_pion_decay(self):
        import naima

        particle_distribution = naima.models.PowerLaw(
            amplitude=2e33 / u.eV, e_0=10 * u.TeV, alpha=2.5
        )
        radiative_model = naima.radiative.PionDecay(
            particle_distribution, nh=1 * u.cm ** -3
        )
        model = NaimaSpectralModel(radiative_model)

        val_at_2TeV = 9.725347355450884e-14 * u.Unit("cm-2 s-1 TeV-1")
        integral_1_10TeV = 3.530537143620737e-13 * u.Unit("cm-2 s-1")
        eflux_1_10TeV = 7.643559573105779e-13 * u.Unit("TeV cm-2 s-1")

        value = model(self.energy)
        assert_quantity_allclose(value, val_at_2TeV)
        assert_quantity_allclose(
            model.integral(emin=self.emin, emax=self.emax), integral_1_10TeV
        )
        assert_quantity_allclose(
            model.energy_flux(emin=self.emin, emax=self.emax), eflux_1_10TeV
        )
        val = model(self.e_array)
        assert val.shape == self.e_array.shape

    def test_ic(self):
        import naima

        particle_distribution = naima.models.ExponentialCutoffBrokenPowerLaw(
            amplitude=2e33 / u.eV,
            e_0=10 * u.TeV,
            alpha_1=2.5,
            alpha_2=2.7,
            e_break=900 * u.GeV,
            e_cutoff=10 * u.TeV,
        )
        radiative_model = naima.radiative.InverseCompton(
            particle_distribution, seed_photon_fields=["CMB"]
        )

        model = NaimaSpectralModel(radiative_model)

        val_at_2TeV = 4.347836316893546e-12 * u.Unit("cm-2 s-1 TeV-1")
        integral_1_10TeV = 1.5958109911918303e-11 * u.Unit("cm-2 s-1")
        eflux_1_10TeV = 2.851281562480875e-11 * u.Unit("TeV cm-2 s-1")

        value = model(self.energy)
        assert_quantity_allclose(value, val_at_2TeV)
        assert_quantity_allclose(
            model.integral(emin=self.emin, emax=self.emax), integral_1_10TeV
        )
        assert_quantity_allclose(
            model.energy_flux(emin=self.emin, emax=self.emax), eflux_1_10TeV
        )
        val = model(self.e_array)
        assert val.shape == self.e_array.shape

    def test_synchrotron(self):
        import naima

        particle_distribution = naima.models.LogParabola(
            amplitude=2e33 / u.eV, e_0=10 * u.TeV, alpha=1.3, beta=0.5
        )
        radiative_model = naima.radiative.Synchrotron(particle_distribution, B=2 * u.G)

        model = NaimaSpectralModel(radiative_model)

        val_at_2TeV = 1.0565840392550432e-24 * u.Unit("cm-2 s-1 TeV-1")
        integral_1_10TeV = 4.4491861907713736e-13 * u.Unit("cm-2 s-1")
        eflux_1_10TeV = 4.594120986691428e-13 * u.Unit("TeV cm-2 s-1")

        value = model(self.energy)
        assert_quantity_allclose(value, val_at_2TeV)
        assert_quantity_allclose(
            model.integral(emin=self.emin, emax=self.emax), integral_1_10TeV
        )
        assert_quantity_allclose(
            model.energy_flux(emin=self.emin, emax=self.emax), eflux_1_10TeV
        )
        val = model(self.e_array)
        assert val.shape == self.e_array.shape

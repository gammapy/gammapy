# Licensed under a 3-clause BSD style license - see LICENSE.rst
import operator
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.maps import MapAxis
from gammapy.modeling.models import (
    SPECTRAL_MODEL_REGISTRY,
    BrokenPowerLawSpectralModel,
    CompoundSpectralModel,
    ConstantSpectralModel,
    EBLAbsorptionNormSpectralModel,
    ExpCutoffPowerLaw3FGLSpectralModel,
    ExpCutoffPowerLawNormSpectralModel,
    ExpCutoffPowerLawSpectralModel,
    GaussianSpectralModel,
    LogParabolaNormSpectralModel,
    LogParabolaSpectralModel,
    NaimaSpectralModel,
    PiecewiseNormSpectralModel,
    PowerLaw2SpectralModel,
    PowerLawNormSpectralModel,
    PowerLawSpectralModel,
    SmoothBrokenPowerLawSpectralModel,
    SuperExpCutoffPowerLaw4FGLSpectralModel,
    TemplateSpectralModel,
)
from gammapy.utils.testing import (
    assert_quantity_allclose,
    mpl_plot_check,
    requires_data,
    requires_dependency,
)


def table_model():
    energy = MapAxis.from_energy_bounds(0.1 * u.TeV, 100 * u.TeV, 1000).center

    model = PowerLawSpectralModel(
        index=2.3, amplitude="4 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    dnde = model(energy)

    return TemplateSpectralModel(energy, dnde)


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
        name="norm-powerlaw",
        model=PowerLawNormSpectralModel(
            tilt=2 * u.Unit(""), norm=4.0 * u.Unit(""), reference=1 * u.TeV,
        ),
        val_at_2TeV=u.Quantity(1.0, ""),
        integral_1_10TeV=u.Quantity(3.6, "TeV"),
        eflux_1_10TeV=u.Quantity(9.210340371976184, "TeV2"),
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
        name="norm-ecpl",
        model=ExpCutoffPowerLawNormSpectralModel(
            index=1.6 * u.Unit(""),
            norm=4 * u.Unit(""),
            reference=1 * u.TeV,
            lambda_=0.1 / u.TeV,
        ),
        val_at_2TeV=u.Quantity(1.080321705479446, ""),
        integral_1_10TeV=u.Quantity(3.765838739678921, "TeV"),
        eflux_1_10TeV=u.Quantity(9.901735870666526, "TeV2"),
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
        name="norm-logpar",
        model=LogParabolaNormSpectralModel(
            alpha=2.3 * u.Unit(""),
            norm=4 * u.Unit(""),
            reference=1 * u.TeV,
            beta=0.5 * u.Unit(""),
        ),
        val_at_2TeV=u.Quantity(0.6387956571420305, ""),
        integral_1_10TeV=u.Quantity(2.255689748270628, "TeV"),
        eflux_1_10TeV=u.Quantity(3.9586515834989267, "TeV2"),
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
        name="ecpl",
        model=ExpCutoffPowerLawSpectralModel(
            index=1.8 * u.Unit(""),
            amplitude=4 / u.cm ** 2 / u.s / u.TeV,
            reference=1 * u.TeV,
            lambda_=0.1 / u.TeV,
            alpha=0.8,
        ),
        val_at_2TeV=u.Quantity(0.871694294554192, "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(3.026342, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(7.38652453, "TeV cm-2 s-1"),
        e_peak=1.7677669529663684 * u.TeV,
    ),
    dict(
        name="bpl",
        model=BrokenPowerLawSpectralModel(
            index1=1.5 * u.Unit(""),
            index2=2.5 * u.Unit(""),
            amplitude=4 / u.cm ** 2 / u.s / u.TeV,
            ebreak=0.5 * u.TeV,
        ),
        val_at_2TeV=u.Quantity(0.125, "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(0.45649740094103286, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(0.9669999668731384, "TeV cm-2 s-1"),
    ),
    dict(
        name="sbpl",
        model=SmoothBrokenPowerLawSpectralModel(
            index1=1.5 * u.Unit(""),
            index2=2.5 * u.Unit(""),
            amplitude=4 / u.cm ** 2 / u.s / u.TeV,
            ebreak=0.5 * u.TeV,
            reference=1 * u.TeV,
            beta=1,
        ),
        val_at_2TeV=u.Quantity(0.28284271247461906, "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(0.9956923907948155, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(2.2372256145972207, "TeV cm-2 s-1"),
    ),
    dict(
        name="sbpl-hard",
        model=SmoothBrokenPowerLawSpectralModel(
            index1=2.5 * u.Unit(""),
            index2=1.5 * u.Unit(""),
            amplitude=4 / u.cm ** 2 / u.s / u.TeV,
            ebreak=0.5 * u.TeV,
            reference=1 * u.TeV,
            beta=1,
        ),
        val_at_2TeV=u.Quantity(3.5355339059327378, "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(13.522782989735022, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(40.06681812966845, "TeV cm-2 s-1"),
    ),
    dict(
        name="pbpl",
        model=PiecewiseNormSpectralModel(
            energy=[1, 3, 7, 10] * u.TeV, norms=[1, 5, 3, 0.5] * u.Unit(""),
        ),
        val_at_2TeV=u.Quantity(2.76058404, ""),
        integral_1_10TeV=u.Quantity(24.758255, "TeV"),
        eflux_1_10TeV=u.Quantity(117.745068, "TeV2"),
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
        name="compound6",
        model=TEST_MODELS[11]["model"] + u.Quantity(4, "cm-2 s-1 TeV-1"),
        val_at_2TeV=TEST_MODELS[11]["val_at_2TeV"] * 2,
        integral_1_10TeV=TEST_MODELS[11]["integral_1_10TeV"] * 2,
        eflux_1_10TeV=TEST_MODELS[11]["eflux_1_10TeV"] * 2,
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


@requires_dependency("scipy")
@pytest.mark.parametrize("spectrum", TEST_MODELS, ids=lambda _: _["name"])
def test_models(spectrum):
    model = spectrum["model"]
    energy = 2 * u.TeV
    value = model(energy)
    assert_quantity_allclose(value, spectrum["val_at_2TeV"], rtol=1e-7)
    if "val_at_3TeV" in spectrum:
        energy = 3 * u.TeV
        value = model(energy)
        assert_quantity_allclose(value, spectrum["val_at_3TeV"], rtol=1e-7)

    energy_min = 1 * u.TeV
    energy_max = 10 * u.TeV
    assert_quantity_allclose(
        model.integral(energy_min=energy_min, energy_max=energy_max),
        spectrum["integral_1_10TeV"],
        rtol=1e-5,
    )
    assert_quantity_allclose(
        model.energy_flux(energy_min=energy_min, energy_max=energy_max),
        spectrum["eflux_1_10TeV"],
        rtol=1e-5,
    )

    if "e_peak" in spectrum:
        assert_quantity_allclose(model.e_peak, spectrum["e_peak"], rtol=1e-2)

    # inverse for ConstantSpectralModel is irrelevant.
    # inverse for Gaussian and PiecewiseNormSpectralModel have a degeneracy
    if not (
        isinstance(model, ConstantSpectralModel)
        or spectrum["name"] == "compound6"
        or spectrum["name"] == "GaussianSpectralModel"
        or spectrum["name"] == "pbpl"
    ):
        assert_quantity_allclose(model.inverse(value), 2 * u.TeV, rtol=0.01)

    if "integral_infinity" in spectrum:
        energy_min = 0 * u.TeV
        energy_max = 10000 * u.TeV
        assert_quantity_allclose(
            model.integral(energy_min=energy_min, energy_max=energy_max),
            spectrum["integral_infinity"],
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
def test_model_plot():
    pwl = PowerLawSpectralModel(
        amplitude=1e-12 * u.Unit("TeV-1 cm-2 s-1"), reference=1 * u.Unit("TeV"), index=2
    )
    pwl.amplitude.error = 0.1e-12 * u.Unit("TeV-1 cm-2 s-1")

    with mpl_plot_check():
        pwl.plot((1 * u.TeV, 10 * u.TeV))

    with mpl_plot_check():
        pwl.plot_error((1 * u.TeV, 10 * u.TeV))


def test_to_from_dict():
    spectrum = TEST_MODELS[0]
    model = spectrum["model"]

    model_dict = model.to_dict()
    model_class = SPECTRAL_MODEL_REGISTRY.get_cls(model_dict["type"])
    new_model = model_class.from_dict(model_dict)

    assert isinstance(new_model, PowerLawSpectralModel)

    actual = [par.value for par in new_model.parameters]
    desired = [par.value for par in model.parameters]
    assert_quantity_allclose(actual, desired)


def test_to_from_dict_compound():
    spectrum = TEST_MODELS[-2]
    model = spectrum["model"]
    assert spectrum["name"] == "compound6"
    model_dict = model.to_dict()
    assert model_dict["operator"] == "add"
    model_class = SPECTRAL_MODEL_REGISTRY.get_cls(model_dict["type"])
    new_model = model_class.from_dict(model_dict)

    assert isinstance(new_model, CompoundSpectralModel)

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
    absorption = EBLAbsorptionNormSpectralModel.read_builtin(
        "dominguez", redshift=redshift
    )

    # Spectral model corresponding to PKS 2155-304 (quiescent state)
    index = 3.53
    amplitude = 1.81 * 1e-12 * u.Unit("cm-2 s-1 TeV-1")
    reference = 1 * u.TeV
    pwl = PowerLawSpectralModel(index=index, amplitude=amplitude, reference=reference)

    # EBL + PWL model
    model = pwl * absorption
    desired = u.Quantity(5.140765e-13, "TeV-1 s-1 cm-2")
    assert_quantity_allclose(model(1 * u.TeV), desired, rtol=1e-3)
    assert model.model2.alpha_norm.value == 1.0

    # EBL + PWL model: test if norm of EBL=0: it mean model =pwl
    model.parameters["alpha_norm"].value = 0
    assert_quantity_allclose(model(1 * u.TeV), pwl(1 * u.TeV), rtol=1e-3)

    # EBL + PWL model: Test with a norm different of 1
    absorption = EBLAbsorptionNormSpectralModel.read_builtin(
        "dominguez", redshift=redshift, alpha_norm=1.5
    )
    model = pwl * absorption
    desired = u.Quantity(2.739695e-13, "TeV-1 s-1 cm-2")
    assert model.model2.alpha_norm.value == 1.5
    assert_quantity_allclose(model(1 * u.TeV), desired, rtol=1e-3)

    # Test error propagation
    model.model1.amplitude.error = 0.1 * model.model1.amplitude.value
    dnde, dnde_err = model.evaluate_error(1 * u.TeV)
    assert_allclose(dnde_err / dnde, 0.1)


@requires_data()
def test_absorbed_extrapolate():
    ebl_model = "dominguez"
    z = 0.0001
    alpha_norm = 1
    absorption = EBLAbsorptionNormSpectralModel.read_builtin(ebl_model)

    values = absorption.evaluate(1 * u.TeV, z, alpha_norm)
    assert_allclose(values, 1)


def test_ecpl_integrate():
    # regression test to check the numerical integration for small energy bins
    ecpl = ExpCutoffPowerLawSpectralModel()
    value = ecpl.integral(1 * u.TeV, 1.1 * u.TeV)
    assert value.isscalar
    assert_quantity_allclose(value, 8.380714e-14 * u.Unit("s-1 cm-2"))


def test_pwl_pivot_energy():
    pwl = PowerLawSpectralModel(amplitude="5.35510540e-11 cm-2 s-1 TeV-1")

    pwl.covariance = [
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
    result = model.evaluate(energy)
    tiny = np.finfo(np.float32).tiny
    mask = abs(values) - tiny > tiny
    np.testing.assert_allclose(
        values[mask] / values.max(), result[mask].value / values.max()
    )
    mask = abs(result.value) - tiny <= tiny
    assert np.all(result[mask] == 0.0)


def test_TemplateSpectralModel_compound():
    energy = [1.00e06, 1.25e06, 1.58e06, 1.99e06] * u.MeV
    values = [4.39e-7, 1.96e-7, 8.80e-7, 3.94e-7] * u.Unit("MeV-1 s-1 sr-1")

    template = TemplateSpectralModel(energy=energy, values=values)
    correction = PowerLawNormSpectralModel(norm=2)
    model = CompoundSpectralModel(template, correction, operator=operator.mul)
    assert np.allclose(model(energy), 2 * values)

    model_mul = template * correction
    assert isinstance(model_mul, CompoundSpectralModel)
    assert np.allclose(model_mul(energy), 2 * values)

    model_dict = model.to_dict()
    assert model_dict["operator"] == "mul"
    model_class = SPECTRAL_MODEL_REGISTRY.get_cls(model_dict["type"])
    new_model = model_class.from_dict(model_dict)
    assert isinstance(new_model, CompoundSpectralModel)
    assert np.allclose(new_model(energy), 2 * values)


@requires_dependency("naima")
class TestNaimaModel:
    # Used to test model value at 2 TeV
    energy = 2 * u.TeV

    # Used to test model integral and energy flux
    energy_min = 1 * u.TeV
    energy_max = 10 * u.TeV

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
            model.integral(energy_min=self.energy_min, energy_max=self.energy_max),
            integral_1_10TeV,
        )
        assert_quantity_allclose(
            model.energy_flux(energy_min=self.energy_min, energy_max=self.energy_max),
            eflux_1_10TeV,
        )
        val = model(self.e_array)
        assert val.shape == self.e_array.shape

        model.amplitude.error = 0.1 * model.amplitude.value

        out = model.evaluate_error(1 * u.TeV)
        assert_allclose(out.data, [5.266068e-13, 5.266068e-14], rtol=1e-3)

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
        integral_1_10TeV = 1.595813e-11 * u.Unit("cm-2 s-1")
        eflux_1_10TeV = 2.851283e-11 * u.Unit("TeV cm-2 s-1")

        value = model(self.energy)
        assert_quantity_allclose(value, val_at_2TeV)
        assert_quantity_allclose(
            model.integral(energy_min=self.energy_min, energy_max=self.energy_max),
            integral_1_10TeV,
            rtol=1e-5,
        )
        assert_quantity_allclose(
            model.energy_flux(energy_min=self.energy_min, energy_max=self.energy_max),
            eflux_1_10TeV,
            rtol=1e-5,
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
        integral_1_10TeV = 4.449186e-13 * u.Unit("cm-2 s-1")
        eflux_1_10TeV = 4.594121e-13 * u.Unit("TeV cm-2 s-1")

        value = model(self.energy)
        assert_quantity_allclose(value, val_at_2TeV)
        assert_quantity_allclose(
            model.integral(energy_min=self.energy_min, energy_max=self.energy_max),
            integral_1_10TeV,
            rtol=1e-5,
        )
        assert_quantity_allclose(
            model.energy_flux(energy_min=self.energy_min, energy_max=self.energy_max),
            eflux_1_10TeV,
            rtol=1e-5,
        )
        val = model(self.e_array)
        assert val.shape == self.e_array.shape

        model.B.value = 3  # update B
        val_at_2TeV = 5.1985064062296e-16 * u.Unit("cm-2 s-1 TeV-1")
        value = model(self.energy)
        assert_quantity_allclose(value, val_at_2TeV)

    def test_ssc(self):
        import naima

        ECBPL = naima.models.ExponentialCutoffBrokenPowerLaw(
            amplitude=3.699e36 / u.eV,
            e_0=1 * u.TeV,
            e_break=0.265 * u.TeV,
            alpha_1=1.5,
            alpha_2=3.233,
            e_cutoff=1863 * u.TeV,
            beta=2.0,
        )

        radiative_model = naima.radiative.InverseCompton(
            ECBPL,
            seed_photon_fields=[
                "CMB",
                ["FIR", 70 * u.K, 0.5 * u.eV / u.cm ** 3],
                ["NIR", 5000 * u.K, 1 * u.eV / u.cm ** 3],
            ],
            Eemax=50 * u.PeV,
            Eemin=0.1 * u.GeV,
        )
        B = 125 * u.uG
        radius = 2.1 * u.pc
        nested_models = {"SSC": {"B": B, "radius": radius}}
        model = NaimaSpectralModel(radiative_model, nested_models=nested_models)
        assert_quantity_allclose(model.B.quantity, B)
        assert_quantity_allclose(model.radius.quantity, radius)
        val_at_2TeV = 1.6703761561806372e-11 * u.Unit("cm-2 s-1 TeV-1")
        value = model(self.energy)
        assert_quantity_allclose(value, val_at_2TeV, rtol=1e-5)

        model.parameters["B"].value = 100
        val_at_2TeV = 1.441331153167876e-11 * u.Unit("cm-2 s-1 TeV-1")
        value = model(self.energy)
        assert_quantity_allclose(value, val_at_2TeV, rtol=1e-5)

    def test_bad_init(self):
        import naima

        particle_distribution = naima.models.PowerLaw(
            amplitude=2e33 / u.eV, e_0=10 * u.TeV, alpha=2.5
        )
        radiative_model = naima.radiative.PionDecay(
            particle_distribution, nh=1 * u.cm ** -3
        )
        model = NaimaSpectralModel(radiative_model)

        with pytest.raises(NotImplementedError):
            NaimaSpectralModel.from_dict(model.to_dict())
        with pytest.raises(NotImplementedError):
            NaimaSpectralModel.from_parameters(model.parameters)


class TestSpectralModelErrorPropagation:
    """Test spectral model error propagation.

    https://github.com/gammapy/gammapy/blob/master/docs/development/pigs/pig-014.rst#proposal
    https://nbviewer.jupyter.org/github/gammapy/gammapy-extra/blob/master/experiments/uncertainty_estimation_prototype.ipynb
    """

    def setup(self):
        self.model = LogParabolaSpectralModel(
            amplitude=3.76e-11 * u.Unit("cm-2 s-1 TeV-1"),
            reference=1 * u.TeV,
            alpha=2.44,
            beta=0.25,
        )
        self.model.covariance = [
            [1.31e-23, 0, -6.80e-14, 3.04e-13],
            [0, 0, 0, 0],
            [-6.80e-14, 0, 0.00899, 0.00904],
            [3.04e-13, 0, 0.00904, 0.0284],
        ]

    def test_evaluate_error_scalar(self):
        # evaluate_error on scalar
        out = self.model.evaluate_error(1 * u.TeV)
        assert isinstance(out, u.Quantity)
        assert out.unit == "cm-2 s-1 TeV-1"
        assert out.shape == (2,)
        assert_allclose(out.data, [3.7600e-11, 3.6193e-12], rtol=1e-3)

    def test_evaluate_error_array(self):
        out = self.model.evaluate_error([1, 100] * u.TeV)
        assert out.shape == (2, 2)
        expected = [[3.76e-11, 2.469e-18], [3.619e-12, 9.375e-18]]
        assert_allclose(out.data, expected, rtol=1e-3)

    def test_evaluate_error_unit(self):
        out = self.model.evaluate_error(1e6 * u.MeV)
        assert out.unit == "cm-2 s-1 TeV-1"
        assert_allclose(out.data, [3.760e-11, 3.6193e-12], rtol=1e-3)

    @pytest.mark.xfail(reason="FIXME, do we need this method?")
    def test_integral_error(self):
        out = self.model.integral_error(1 * u.TeV, 10 * u.TeV)
        assert out.unit == "cm-2 s-1"
        assert out.shape == (2,)
        assert_allclose(out.data, [2.197e-11, 2.796e-12], rtol=1e-3)

    @pytest.mark.xfail(reason="FIXME, do we need this method?")
    def test_energy_flux_error(self):
        out = self.model.energy_flux_error(1 * u.TeV, 10 * u.TeV)
        assert out.unit == "TeV cm-2 s-1"
        assert out.shape == (2,)
        assert_allclose(out.data, [4.119e-11, 8.157e-12], rtol=1e-3)

    def test_ecpl_model(self):
        # Regression test for ECPL model
        # https://github.com/gammapy/gammapy/issues/2007
        model = ExpCutoffPowerLawSpectralModel(
            amplitude=2.076183759227292e-12 * u.Unit("cm-2 s-1 TeV-1"),
            index=1.8763343736076483,
            lambda_=0.08703226432146616 * u.Unit("TeV-1"),
            reference=1 * u.TeV,
        )
        model.covariance = [
            [0.00204191498, -1.507724e-14, 0.0, -0.001834819, 0.0],
            [-1.507724e-14, 1.6864740e-25, 0.0, 1.854251e-14, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.001834819175, 1.8542517e-14, 0.0, 0.0032559101, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]

        out = model.evaluate_error(1 * u.TeV)
        assert_allclose(out.data, [1.903129e-12, 2.979976e-13], rtol=1e-3)

        out = model.evaluate_error(0.1 * u.TeV)
        assert_allclose(out.data, [1.548176e-10, 1.933612e-11], rtol=1e-3)


def test_integral_error_PowerLaw():
    energy = np.linspace(1 * u.TeV, 10 * u.TeV, 10)
    energy_min = energy[:-1]
    energy_max = energy[1:]

    powerlaw = PowerLawSpectralModel()
    powerlaw.parameters["index"].error = 0.4
    powerlaw.parameters["amplitude"].error = 1e-13

    flux, flux_error = powerlaw.integral_error(energy_min, energy_max)

    assert_allclose(flux.value[0] / 1e-13, 5.0, rtol=0.1)
    assert_allclose(flux_error.value[0] / 1e-14, 8.546615432273905, rtol=0.01)


def test_integral_error_ExpCutOffPowerLaw():
    energy = np.linspace(1 * u.TeV, 10 * u.TeV, 10)
    energy_min = energy[:-1]
    energy_max = energy[1:]

    exppowerlaw = ExpCutoffPowerLawSpectralModel()
    exppowerlaw.parameters["index"].error = 0.4
    exppowerlaw.parameters["amplitude"].error = 1e-13
    exppowerlaw.parameters["lambda_"].error = 0.03

    flux, flux_error = exppowerlaw.integral_error(energy_min, energy_max)

    assert_allclose(flux.value[0] / 1e-13, 5.05855622, rtol=0.01)
    assert_allclose(flux_error.value[0] / 1e-14, 8.9598, rtol=0.01)


def test_energy_flux_error_power_Law():
    energy_min = 1 * u.TeV
    energy_max = 10 * u.TeV

    powerlaw = PowerLawSpectralModel()
    powerlaw.parameters["index"].error = 0.4
    powerlaw.parameters["amplitude"].error = 1e-13

    enrg_flux, enrg_flux_error = powerlaw.energy_flux_error(energy_min, energy_max)
    assert_allclose(enrg_flux.value / 1e-12, 2.303, rtol=0.001)
    assert_allclose(enrg_flux_error.value / 1e-12, 1.085, rtol=0.001)


def test_energy_flux_error_exp_cutoff_power_law():
    energy_min = 1 * u.TeV
    energy_max = 10 * u.TeV

    exppowerlaw = ExpCutoffPowerLawSpectralModel()
    exppowerlaw.parameters["index"].error = 0.4
    exppowerlaw.parameters["amplitude"].error = 1e-13
    exppowerlaw.parameters["lambda_"].error = 0.03

    enrg_flux, enrg_flux_error = exppowerlaw.energy_flux_error(energy_min, energy_max)

    assert_allclose(enrg_flux.value / 1e-12, 2.788, rtol=0.001)
    assert_allclose(enrg_flux_error.value / 1e-12, 1.419, rtol=0.001)

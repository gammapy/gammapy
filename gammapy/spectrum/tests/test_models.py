# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import astropy.units as u
from ...utils.energy import EnergyBounds
from ...utils.testing import assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data, mpl_plot_check
from ..models import (
    PowerLaw,
    PowerLaw2,
    ExponentialCutoffPowerLaw,
    ExponentialCutoffPowerLaw3FGL,
    LogParabola,
    TableModel,
    AbsorbedSpectralModel,
    Absorption,
    ConstantModel,
)


def table_model():
    energy_edges = EnergyBounds.equal_log_spacing(0.1 * u.TeV, 100 * u.TeV, 1000)
    energy = energy_edges.log_centers

    index = 2.3 * u.Unit("")
    amplitude = 4 / u.cm ** 2 / u.s / u.TeV
    reference = 1 * u.TeV
    pl = PowerLaw(index, amplitude, reference)
    flux = pl(energy)

    return TableModel(energy, flux, 1 * u.Unit(""))


TEST_MODELS = [
    dict(
        name="powerlaw",
        model=PowerLaw(
            index=2.3 * u.Unit(""),
            amplitude=4 / u.cm ** 2 / u.s / u.TeV,
            reference=1 * u.TeV,
        ),
        val_at_2TeV=u.Quantity(4 * 2. ** (-2.3), "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(2.9227116204223784, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(6.650836884969039, "TeV cm-2 s-1"),
    ),
    dict(
        name="powerlaw",
        model=PowerLaw(
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
        model=PowerLaw2(
            amplitude=u.Quantity(2.9227116204223784, "cm-2 s-1"),
            index=2.3 * u.Unit(""),
            emin=1 * u.TeV,
            emax=10 * u.TeV,
        ),
        val_at_2TeV=u.Quantity(4 * 2. ** (-2.3), "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(2.9227116204223784, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(6.650836884969039, "TeV cm-2 s-1"),
    ),
    dict(
        name="ecpl",
        model=ExponentialCutoffPowerLaw(
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
        model=ExponentialCutoffPowerLaw3FGL(
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
        name="logpar",
        model=LogParabola(
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
        model=LogParabola.from_log10(
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
        model=ConstantModel(const=4 / u.cm ** 2 / u.s / u.TeV),
        val_at_2TeV=u.Quantity(4, "cm-2 s-1 TeV-1"),
        integral_1_10TeV=u.Quantity(35.9999999999999, "cm-2 s-1"),
        eflux_1_10TeV=u.Quantity(198.00000000000006, "TeV cm-2 s-1"),
    ),
]

# Add compound models
TEST_MODELS.append(
    dict(
        name="compound1",
        model=TEST_MODELS[0]["model"] * 5,
        val_at_2TeV=TEST_MODELS[0]["val_at_2TeV"] * 5,
        integral_1_10TeV=TEST_MODELS[0]["integral_1_10TeV"] * 5,
        eflux_1_10TeV=TEST_MODELS[0]["eflux_1_10TeV"] * 5,
    )
)

TEST_MODELS.append(
    dict(
        name="compound2",
        model=5 * TEST_MODELS[0]["model"],
        val_at_2TeV=TEST_MODELS[0]["val_at_2TeV"] * 5,
        integral_1_10TeV=TEST_MODELS[0]["integral_1_10TeV"] * 5,
        eflux_1_10TeV=TEST_MODELS[0]["eflux_1_10TeV"] * 5,
    )
)

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
        name="compound5",
        model=TEST_MODELS[0]["model"] - TEST_MODELS[0]["model"] / 2.,
        val_at_2TeV=0.5 * TEST_MODELS[0]["val_at_2TeV"],
        integral_1_10TeV=TEST_MODELS[0]["integral_1_10TeV"] * 0.5,
        eflux_1_10TeV=TEST_MODELS[0]["eflux_1_10TeV"] * 0.5,
    )
)
# The table model imports scipy.interpolate in `__init__`,
# so we skip it if scipy is not available
try:
    TEST_MODELS.append(
        dict(
            name="table_model",
            model=table_model(),
            # Values took from power law expectation
            val_at_2TeV=u.Quantity(4 * 2. ** (-2.3), "cm-2 s-1 TeV-1"),
            integral_1_10TeV=u.Quantity(2.9227116204223784, "cm-2 s-1"),
            eflux_1_10TeV=u.Quantity(6.650836884969039, "TeV cm-2 s-1"),
        )
    )
except ImportError:
    pass


@requires_dependency("uncertainties")
@pytest.mark.parametrize("spectrum", TEST_MODELS, ids=[_["name"] for _ in TEST_MODELS])
def test_models(spectrum):
    model = spectrum["model"]
    energy = 2 * u.TeV
    value = model(energy)
    assert_quantity_allclose(value, spectrum["val_at_2TeV"])
    emin = 1 * u.TeV
    emax = 10 * u.TeV
    assert_quantity_allclose(
        model.integral(emin=emin, emax=emax), spectrum["integral_1_10TeV"]
    )
    assert_quantity_allclose(
        model.energy_flux(emin=emin, emax=emax), spectrum["eflux_1_10TeV"]
    )

    if "e_peak" in spectrum:
        assert_quantity_allclose(model.e_peak, spectrum["e_peak"], rtol=1E-2)

    # inverse for TableModel is not implemented
    if not (isinstance(model, TableModel) or isinstance(model, ConstantModel)):
        assert_quantity_allclose(model.inverse(value), 2 * u.TeV, rtol=0.05)

    model.to_dict()

    assert "" in str(model)

    # check that an array evaluation works (otherwise e.g. plotting raises an error)
    e_array = [2, 10, 20] * u.TeV
    val = model(e_array)
    assert_quantity_allclose(val[0], spectrum["val_at_2TeV"])


@requires_dependency("matplotlib")
@requires_data("gammapy-extra")
def test_table_model_from_file():
    filename = "$GAMMAPY_EXTRA/datasets/ebl/ebl_franceschini.fits.gz"
    absorption_z03 = TableModel.read_xspec_model(filename=filename, param=0.3)
    with mpl_plot_check():
        absorption_z03.plot(energy_range=(0.03, 10), energy_unit=u.TeV, flux_unit="")


@requires_data("gammapy-extra")
@requires_dependency("scipy")
def test_absorption():
    # absorption values for given redshift
    redshift = 0.117
    absorption = Absorption.read_builtin("dominguez")

    # Spectral model corresponding to PKS 2155-304 (quiescent state)
    index = 3.53
    amplitude = 1.81 * 1e-12 * u.Unit("cm-2 s-1 TeV-1")
    reference = 1 * u.TeV
    pwl = PowerLaw(index=index, amplitude=amplitude, reference=reference)

    # EBL + PWL model
    model = AbsorbedSpectralModel(
        spectral_model=pwl, absorption=absorption, parameter=redshift
    )

    # Test if the absorption factor at the reference energy
    # corresponds to the ratio of the absorbed flux
    # divided by the flux of the spectral model
    kwargs = dict(
        index=index, amplitude=amplitude, reference=reference, redshift=redshift
    )
    model_ref_energy = model.evaluate(energy=reference, **kwargs)
    pwl_ref_energy = pwl.evaluate(
        energy=reference, index=index, amplitude=amplitude, reference=reference
    )

    desired = absorption.evaluate(energy=reference, parameter=redshift)
    actual = model_ref_energy / pwl_ref_energy
    assert_quantity_allclose(actual, desired)


@requires_dependency("uncertainties")
def test_pwl_index_2_error():
    pars, errs = {}, {}
    pars["amplitude"] = 1e-12 * u.Unit("TeV-1 cm-2 s-1")
    pars["reference"] = 1 * u.Unit("TeV")
    pars["index"] = 2 * u.Unit("")
    errs["amplitude"] = 0.1e-12 * u.Unit("TeV-1 cm-2 s-1")

    pwl = PowerLaw(**pars)
    pwl.parameters.set_parameter_errors(errs)

    val, val_err = pwl.evaluate_error(1 * u.TeV)
    assert_quantity_allclose(val, 1e-12 * u.Unit("TeV-1 cm-2 s-1"))
    assert_quantity_allclose(val_err, 0.1e-12 * u.Unit("TeV-1 cm-2 s-1"))

    flux, flux_err = pwl.integral_error(1 * u.TeV, 10 * u.TeV)
    assert_quantity_allclose(flux, 9e-13 * u.Unit("cm-2 s-1"))
    assert_quantity_allclose(flux_err, 9e-14 * u.Unit("cm-2 s-1"))

    eflux, eflux_err = pwl.energy_flux_error(1 * u.TeV, 10 * u.TeV)
    assert_quantity_allclose(eflux, 2.302585E-12 * u.Unit("TeV cm-2 s-1"))
    assert_quantity_allclose(eflux_err, 0.2302585E-12 * u.Unit("TeV cm-2 s-1"))


@requires_data("gammapy-extra")
@requires_dependency("scipy")
def test_fermi_isotropic():
    filename = "$GAMMAPY_EXTRA/datasets/fermi_3fhl/iso_P8R2_SOURCE_V6_v06.txt"
    model = TableModel.read_fermi_isotropic_model(filename)
    assert_quantity_allclose(
        model(50 * u.GeV), 1.463 * u.Unit("1e-13 MeV-1 cm-2 s-1 sr-1"), rtol=1e-3
    )

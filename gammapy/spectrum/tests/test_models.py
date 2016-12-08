# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import astropy.units as u
from gammapy.utils.energy import EnergyBounds
from astropy.tests.helper import assert_quantity_allclose, pytest
from ..models import (PowerLaw, PowerLaw2, ExponentialCutoffPowerLaw,
                      ExponentialCutoffPowerLaw3FGL, LogParabola, TableModel)
from ...utils.testing import requires_dependency


def table_model():
    energy_edges = EnergyBounds.equal_log_spacing(0.1 * u.TeV, 100 * u.TeV, 1000)
    energy = energy_edges.log_centers

    index = 2.3 * u.Unit('')
    amplitude = 4 / u.cm ** 2 / u.s / u.TeV
    reference = 1 * u.TeV
    pl = PowerLaw(index, amplitude, reference)
    flux = pl(energy)

    return TableModel(energy, flux, 1 * u.Unit(''))


TEST_MODELS = [
    dict(
        name='powerlaw',
        model=PowerLaw(
            index=2.3 * u.Unit(''),
            amplitude=4 / u.cm ** 2 / u.s / u.TeV,
            reference=1 * u.TeV
        ),
        val_at_2TeV=u.Quantity(4 * 2. ** (-2.3), 'cm-2 s-1 TeV-1'),
        integral_1_10TeV=u.Quantity(2.9227116204223784, 'cm-2 s-1'),
        eflux_1_10TeV=u.Quantity(6.650836884969039, 'TeV cm-2 s-1'),
    ),

    dict(
        name='powerlaw',
        model=PowerLaw(
            index=2 * u.Unit(''),
            amplitude=4 / u.cm ** 2 / u.s / u.TeV,
            reference=1 * u.TeV
        ),
        val_at_2TeV=u.Quantity(1.0, 'cm-2 s-1 TeV-1'),
        integral_1_10TeV=u.Quantity(3.6, 'cm-2 s-1'),
        eflux_1_10TeV=u.Quantity(9.210340371976184, 'TeV cm-2 s-1'),
    ),

    dict(
        name='powerlaw2',
        model=PowerLaw2(
            amplitude=u.Quantity(2.9227116204223784, 'cm-2 s-1'),
            index=2.3 * u.Unit(''),
            emin=1 * u.TeV,
            emax=10 * u.TeV
        ),
        val_at_2TeV=u.Quantity(4 * 2. ** (-2.3), 'cm-2 s-1 TeV-1'),
        integral_1_10TeV=u.Quantity(2.9227116204223784, 'cm-2 s-1'),
        eflux_1_10TeV=u.Quantity(6.650836884969039, 'TeV cm-2 s-1'),
    ),

    dict(
        name='ecpl',
        model=ExponentialCutoffPowerLaw(
            index=2.3 * u.Unit(''),
            amplitude=4 / u.cm ** 2 / u.s / u.TeV,
            reference=1 * u.TeV,
            lambda_=0.1 / u.TeV
        ),

        val_at_2TeV=u.Quantity(0.6650160161581361, 'cm-2 s-1 TeV-1'),
        integral_1_10TeV=u.Quantity(2.3556579120286796, 'cm-2 s-1'),
        eflux_1_10TeV=u.Quantity(4.83209019773561, 'TeV cm-2 s-1'),
    ),

    dict(
        name='ecpl_3fgl',
        model=ExponentialCutoffPowerLaw3FGL(
            index=2.3 * u.Unit(''),
            amplitude=4 / u.cm ** 2 / u.s / u.TeV,
            reference=1 * u.TeV,
            ecut=10 * u.TeV
        ),

        val_at_2TeV=u.Quantity(0.7349563611124971, 'cm-2 s-1 TeV-1'),
        integral_1_10TeV=u.Quantity(2.6034046173089, 'cm-2 s-1'),
        eflux_1_10TeV=u.Quantity(5.340285560055799, 'TeV cm-2 s-1'),
    ),

    dict(
        name='logpar',
        model=LogParabola(
            alpha=2.3 * u.Unit(''),
            amplitude=4 / u.cm ** 2 / u.s / u.TeV,
            reference=1 * u.TeV,
            beta=0.5 * u.Unit('')
        ),

        val_at_2TeV=u.Quantity(0.6387956571420305, 'cm-2 s-1 TeV-1'),
        integral_1_10TeV=u.Quantity(2.255689748270628, 'cm-2 s-1'),
        eflux_1_10TeV=u.Quantity(3.9586515834989267, 'TeV cm-2 s-1')
    ),
]

# The table model imports scipy.interpolate in `__init__`,
# so we skip it if scipy is not available
try:
    TEST_MODELS.append(dict(
        name='table_model',
        model=table_model(),
        # Values took from power law expectation
        val_at_2TeV=u.Quantity(4 * 2. ** (-2.3), 'cm-2 s-1 TeV-1'),
        integral_1_10TeV=u.Quantity(2.9227116204223784, 'cm-2 s-1'),
        eflux_1_10TeV=u.Quantity(6.650836884969039, 'TeV cm-2 s-1'),
    ))
except ImportError:
    pass


@requires_dependency('uncertainties')
@pytest.mark.parametrize(
    "spectrum", TEST_MODELS, ids=[_['name'] for _ in TEST_MODELS]
)
def test_models(spectrum):
    model = spectrum['model']
    energy = 2 * u.TeV
    value = model(energy)
    assert_quantity_allclose(value, spectrum['val_at_2TeV'])
    emin = 1 * u.TeV
    emax = 10 * u.TeV
    assert_quantity_allclose(model.integral(emin=emin, emax=emax),
                             spectrum['integral_1_10TeV'])
    assert_quantity_allclose(model.energy_flux(emin=emin, emax=emax),
                             spectrum['eflux_1_10TeV'])

    # inverse for TableModel is not implemented
    if not isinstance(model, TableModel):
        assert_quantity_allclose(model.inverse(value), 2 * u.TeV, rtol=0.05)
    model.to_dict()


@requires_dependency('matplotlib')
@requires_dependency('sherpa')
@pytest.mark.parametrize(
    "spectrum", TEST_MODELS, ids=[_['name'] for _ in TEST_MODELS]
)
def test_to_sherpa(spectrum):
    model = spectrum['model']
    try:
        sherpa_model = model.to_sherpa()
    except NotImplementedError:
        pass
    else:
        test_e = 1.56 * u.TeV
        desired = model(test_e)
        actual = sherpa_model(test_e.to('keV').value) * u.Unit('cm-2 s-1 keV-1')
        assert_quantity_allclose(actual, desired)

    # test plot
    energy_range = [1, 10] * u.TeV
    model.plot(energy_range=energy_range)

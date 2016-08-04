# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose, pytest
from ..models import PowerLaw, ExponentialCutoffPowerLaw, LogParabola
from ...utils.testing import requires_dependency

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


@pytest.mark.parametrize(
    "spectrum", TEST_MODELS, ids=[_.name for _ in TEST_MODELS]
)
def test_models(spectrum):
    model = spectrum['model']
    energy = 2 * u.TeV
    assert_quantity_allclose(model(energy), spectrum['val_at_2TeV'])
    emin = 1 * u.TeV
    emax = 10 * u.TeV
    assert_quantity_allclose(model.integral(emin=emin, emax=emax),
                             spectrum['integral_1_10TeV'])
    assert_quantity_allclose(model.energy_flux(emin=emin, emax=emax),
                             spectrum['eflux_1_10TeV'])
    model.to_dict()


@requires_dependency('matplotlib')
@requires_dependency('sherpa')
@pytest.mark.parametrize("spectrum", TEST_MODELS.values(), ids=TEST_MODELS.keys())
def test_to_sherpa(spectrum):
    model = spectrum['model']
    try:
        model.to_sherpa()
    except AttributeError:
        pass
    energy_range = [1, 10] * u.TeV
    model.plot(energy_range=energy_range)

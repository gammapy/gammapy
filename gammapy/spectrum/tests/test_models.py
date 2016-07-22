# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ..models import PowerLaw, ExponentialCutoffPowerLaw
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose, pytest
from ...utils.testing import requires_dependency


def get_test_data():
    data = list()
    powerlaw = list()
    powerlaw.append(PowerLaw(index=2.3 * u.Unit(''),
                             amplitude=4 / u.cm ** 2 / u.s / u.TeV,
                             reference = 1 * u.TeV))

    powerlaw.append(dict(val_at_2TeV=u.Quantity(4 * 2. ** (-2.3), 'cm-2 s-1 TeV-1'),
                         integral_1_10TeV=u.Quantity(2.9227116204223784, 'cm-2 s-1'),
                         eflux_1_10TeV=u.Quantity(6.650836884969039, 'TeV cm-2 s-1')))
    
    ecpl = list()
    ecpl.append(ExponentialCutoffPowerLaw(index=2.3 * u.Unit(''),
                                          amplitude=4 / u.cm ** 2 / u.s / u.TeV,
                                          reference = 1 * u.TeV,
                                          lambda_=0.1 / u.TeV))

    ecpl.append(dict(val_at_2TeV=u.Quantity(0.6650160161581361, 'cm-2 s-1 TeV-1'),
                     integral_1_10TeV=u.Quantity(2.3556579120286796, 'cm-2 s-1'),
                     eflux_1_10TeV=u.Quantity(4.83209019773561, 'TeV cm-2 s-1')))
    
    data.append(powerlaw)
    data.append(ecpl)
    return data


@pytest.mark.parametrize("model, results", get_test_data())
def test_models(model, results):
    energy = 2 * u.TeV
    assert_quantity_allclose(model(energy), results['val_at_2TeV'])
    emin = 1 * u.TeV
    emax = 10 * u.TeV
    assert_quantity_allclose(model.integral(emin=emin, emax=emax),
                             results['integral_1_10TeV'])
    assert_quantity_allclose(model.energy_flux(emin=emin, emax=emax),
                             results['eflux_1_10TeV'])
    model.to_dict()


@requires_dependency('matplotlib')
@requires_dependency('sherpa')
@pytest.mark.parametrize("model, results", get_test_data())
def test_to_sherpa(model, results):
    try:
        model.to_sherpa()
    except AttributeError:
        pass
    energy_range = [1,10] * u.TeV
    model.plot(energy_range = energy_range)

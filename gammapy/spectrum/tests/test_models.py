# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ..models import PowerLaw, ExponentialCutoffPowerLaw
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose, pytest


def get_test_data():
    data = list()
    powerlaw = list()
    powerlaw.append(PowerLaw(index=2 * u.Unit(''),
                             amplitude=4 / u.cm ** 2 / u.s / u.TeV,
                             reference = 1 * u.TeV))

    powerlaw.append(dict(val_at_2TeV=u.Quantity(1, 'cm-2 s-1 TeV-1'),
                        integral_1_10TeV=u.Quantity(3.6, 'cm-2 s-1')))
    
    data.append(powerlaw)
    return data


@pytest.mark.parametrize("model, results", get_test_data())
def test_models(model, results):
    energy = 2 * u.TeV
    assert_quantity_allclose(model(energy), results['val_at_2TeV'])
    emin = 1 * u.TeV
    emax = 10 * u.TeV
    assert_quantity_allclose(model.integral(emin=emin, emax=emax),
                             results['integral_1_10TeV'])
    
    


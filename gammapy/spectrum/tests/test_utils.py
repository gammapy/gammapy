# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.units import Quantity
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
import pytest
from ...utils.testing import requires_dependency
from ...irf import EffectiveAreaTable, EnergyDispersion
from ...spectrum import LogEnergyAxis, integrate_spectrum, CountsPredictor
from ..powerlaw import power_law_energy_flux, power_law_evaluate, power_law_flux
from ..models import ExponentialCutoffPowerLaw, PowerLaw, TableModel


@requires_dependency('scipy')
def test_LogEnergyAxis():
    from scipy.stats import gmean
    energy = Quantity([1, 10, 100], 'TeV')
    energy_axis = LogEnergyAxis(energy)

    energy = Quantity(gmean([1, 10]), 'TeV')
    pix = energy_axis.wcs_world2pix(energy.to('MeV'))
    assert_allclose(pix, 0.5)

    world = energy_axis.wcs_pix2world(pix)
    assert_quantity_allclose(world, energy)


def test_integrate_spectrum():
    """
    Test numerical integration against analytical solution.
    """
    e1 = Quantity(1, 'TeV')
    e2 = Quantity(10, 'TeV')
    einf = Quantity(1e10, 'TeV')
    e = Quantity(1, 'TeV')
    g = 2.3
    I = Quantity(1e-12, 'cm-2 s-1')

    ref = power_law_energy_flux(I=I, g=g, e=e, e1=e1, e2=e2)
    norm = power_law_flux(I=I, g=g, e=e, e1=e1, e2=einf)
    f = lambda x: x * power_law_evaluate(x, norm, g, e)
    val = integrate_spectrum(f, e1, e2)
    assert_quantity_allclose(val, ref)

    # Test quantity handling
    e2_ = Quantity(1e4, 'GeV')
    val_ = integrate_spectrum(f, e1, e2_)
    assert_quantity_allclose(val, val_)


@requires_dependency('uncertainties')
def test_integrate_spectrum_uncertainties():
    """
    Test numerical integration against analytical solution.
    """
    from uncertainties import unumpy
    e1 = 1.
    e2 = 10.
    einf = 1e10
    e = 1.
    g = unumpy.uarray(2.3, 0.2)
    I = unumpy.uarray(1e-12, 1e-13)

    ref = power_law_energy_flux(I=I, g=g, e=e, e1=e1, e2=e2)
    norm = power_law_flux(I=I, g=g, e=e, e1=e1, e2=einf)
    f = lambda x: x * power_law_evaluate(x, norm, g, e)
    val = integrate_spectrum(f, e1, e2)

    assert_allclose(unumpy.nominal_values(val), unumpy.nominal_values(ref))
    assert_allclose(unumpy.std_devs(val), unumpy.std_devs(ref))


@pytest.mark.xfail(reason='Spectral models cannot handle ufuncs properly')
@requires_dependency('uncertainties')
def test_integrate_spectrum_ecpl():
    """
    Test ecpl integration. Regression test for
    https://github.com/gammapy/gammapy/issues/687
    """
    from uncertainties import unumpy
    amplitude = unumpy.uarray(1e-12, 1e-13)
    index = unumpy.uarray(2.3, 0.2)
    reference = 1
    lambda_ = 0.1
    ecpl = ExponentialCutoffPowerLaw(index, amplitude, reference, lambda_)
    emin, emax = 1, 1e10
    val = ecpl.integral(emin, emax)

    assert_allclose(unumpy.nominal_values(val), 5.956578235358054e-13)
    assert_allclose(unumpy.std_devs(val), 9.278302514378108e-14)


def get_test_cases():
    e_true = Quantity(np.logspace(-1, 2, 120), 'TeV')
    e_reco = Quantity(np.logspace(-1, 2, 100), 'TeV')

    try:
        import scipy
    except ImportError:
        return []
    else:
        return [
            dict(model=PowerLaw(index=2,
                                reference=Quantity(1, 'TeV'),
                                amplitude=Quantity(1e2, 'TeV-1')),
                 e_true=e_true,
                 npred=999),
            dict(model=PowerLaw(index=2,
                                reference=Quantity(1, 'TeV'),
                                amplitude=Quantity(1e-11, 'TeV-1 cm-2 s-1')),
                 aeff=EffectiveAreaTable.from_parametrization(e_true),
                 livetime=Quantity(10, 'h'),
                 npred=1448.059605038253),
            dict(model=PowerLaw(index=2,
                                reference=Quantity(1, 'GeV'),
                                amplitude=Quantity(1e-11, 'GeV-1 cm-2 s-1')),
                 aeff=EffectiveAreaTable.from_parametrization(e_true),
                 livetime=Quantity(30, 'h'),
                 npred=4.344178815114759),
            dict(model=PowerLaw(index=2,
                                reference=Quantity(1, 'TeV'),
                                amplitude=Quantity(1e-11, 'TeV-1 cm-2 s-1')),
                 aeff=EffectiveAreaTable.from_parametrization(e_true),
                 edisp=EnergyDispersion.from_gauss(e_reco=e_reco,
                                                   e_true=e_true,
                                                   bias=0, sigma=0.2),
                 livetime=Quantity(10, 'h'),
                 npred=1437.4542016322125),
            dict(model=TableModel(energy=[0.1, 0.2, 0.3, 0.4] * u.TeV,
                                  values=[4., 3., 1., 0.1] * u.Unit('TeV-1')),
                 npred=0.5545130625383198,
                 e_true=[0.1, 0.2, 0.3, 0.4] * u.TeV)
        ]


@requires_dependency('scipy')
@pytest.mark.parametrize('case', get_test_cases())
def test_counts_predictor(case):
    desired = case.pop('npred')
    predictor = CountsPredictor(**case)
    predictor.run()
    actual = predictor.npred.total_counts.value
    assert_allclose(actual, desired)

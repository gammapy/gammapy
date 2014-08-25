# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_allclose
import numpy as np
from astropy.table import Table
from astropy.units import Quantity
from astropy.tests.helper import pytest
from ....datasets import load_atnf_sample
from ...source import Pulsar, SimplePulsar

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

pulsar = Pulsar()
t = Quantity([1E2, 1E4, 1E6, 1E8], 'yr')


def test_SimplePulsar_atnf():
    """Test functions against ATNF pulsar catalog values"""
    atnf = load_atnf_sample()
    P = Quantity(atnf['P0'], 's')
    P_dot = Quantity(atnf['P1'], '')
    simple_pulsar = SimplePulsar(P=P, P_dot=P_dot)
    assert_allclose(simple_pulsar.tau.to('yr'), atnf['AGE'], rtol=0.01)
    assert_allclose(simple_pulsar.luminosity_spindown.to('erg s^-1'), atnf['EDOT'], rtol=0.01)
    assert_allclose(simple_pulsar.magnetic_field.to('gauss'), atnf['BSURF'], rtol=0.01)


def test_Pulsar_period():
    """Test pulsar period"""
    reference = [0.10000001, 0.10000123, 0.10012331, 0.11270709]
    assert_allclose(pulsar.period(t), reference)


def test_Pulsar_peridod_dot():
    """Test pulsar period derivative"""
    reference = [9.76562380e-19, 9.76550462e-19, 9.75359785e-19, 8.66460603e-19]
    assert_allclose(pulsar.period_dot(t), reference)


def test_Pulsar_luminosity_spindown():
    """Test pulsar spin down luminosity"""
    reference = [3.85531469e+31, 3.85536174e+31, 3.86006820e+31, 4.34521233e+31]
    assert_allclose(pulsar.luminosity_spindown(t), reference)


@pytest.mark.skipif('not HAS_SCIPY')
def test_Pulsar_energy_integrated():
    """Test against numerical integration"""
    energies = []
    from scipy.integrate import quad

    def lumi(t):
        t = Quantity(t, 's')
        return pulsar.luminosity_spindown(t).value

    for t_ in t:
        energy = quad(lumi, 0, t_.cgs.value, epsrel=0.01)[0]
        energies.append(energy)
    # The last value is quite inaccurate, because integration is over
    # several decades
    assert_allclose(energies, pulsar.energy_integrated(t), rtol=0.2)


def test_Pulsar_magnetic_field():
    """Test against numerical integration"""
    reference = np.ones_like(t) * 10 ** pulsar.logB
    assert_allclose(pulsar.magnetic_field(t), reference)

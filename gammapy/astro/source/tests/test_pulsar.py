# Licensed under a 3-clause BSD style license - see LICENSE.rst
import scipy.integrate
from numpy.testing import assert_allclose
from astropy.table import Table
from astropy.units import Quantity
from gammapy.astro.source import Pulsar, SimplePulsar
from gammapy.utils.testing import assert_quantity_allclose

pulsar = Pulsar()
time = Quantity([1e2, 1e4, 1e6, 1e8], "yr")


def get_atnf_catalog_sample():
    data = """
    NUM   NAME            Gl          Gb       P0        P1        AGE        BSURF     EDOT
    1     J0006+1834      108.172    -42.985   0.693748  2.10e-15  5.24e+06   1.22e+12  2.48e+32
    2     J0007+7303      119.660     10.463   0.315873  3.60e-13  1.39e+04   1.08e+13  4.51e+35
    3     B0011+47        116.497    -14.631   1.240699  5.64e-16  3.48e+07   8.47e+11  1.17e+31
    7     B0021-72E       305.883    -44.883   0.003536  9.85e-20  5.69e+08   5.97e+08  8.79e+34
    8     B0021-72F       305.899    -44.892   0.002624  6.45e-20  6.44e+08   4.16e+08  1.41e+35
    16    J0024-7204O     305.897    -44.889   0.002643  3.04e-20  1.38e+09   2.87e+08  6.49e+34
    18    J0024-7204Q     305.877    -44.899   0.004033  3.40e-20  1.88e+09   3.75e+08  2.05e+34
    21    J0024-7204T     305.890    -44.894   0.007588  2.94e-19  4.09e+08   1.51e+09  2.65e+34
    22    J0024-7204U     305.890    -44.905   0.004343  9.52e-20  7.23e+08   6.51e+08  4.59e+34
    28    J0026+6320      120.176      0.593   0.318358  1.50e-16  3.36e+07   2.21e+11  1.84e+32
    """
    return Table.read(data, format="ascii")


def test_SimplePulsar_atnf():
    """Test functions against ATNF pulsar catalog values"""
    atnf = get_atnf_catalog_sample()
    simple_pulsar = SimplePulsar(
        P=Quantity(atnf["P0"], "s"), P_dot=Quantity(atnf["P1"], "")
    )

    assert_allclose(simple_pulsar.tau.to("yr").value, atnf["AGE"], rtol=0.01)

    edot = simple_pulsar.luminosity_spindown.to("erg s^-1").value
    assert_allclose(edot, atnf["EDOT"], rtol=0.01)

    bsurf = simple_pulsar.magnetic_field.to("gauss").value
    assert_allclose(bsurf, atnf["BSURF"], rtol=0.01)


def test_Pulsar_period():
    """Test pulsar period"""
    reference = Quantity([0.1, 0.10000031, 0.10003081, 0.10303572], "s")
    assert_quantity_allclose(pulsar.period(time), reference)


def test_Pulsar_peridod_dot():
    """Test pulsar period derivative"""
    reference = [9.76562470e-19, 9.76559490e-19, 9.76261682e-19, 9.47790252e-19]
    assert_allclose(pulsar.period_dot(time), reference)


def test_Pulsar_luminosity_spindown():
    """Test pulsar spin down luminosity"""
    reference = [3.85531374e31, 3.85526669e31, 3.85056609e31, 3.42064935e31]
    assert_allclose(pulsar.luminosity_spindown(time).value, reference)


def test_Pulsar_energy_integrated():
    """Test against numerical integration"""
    energies = []

    def lumi(t):
        t = Quantity(t, "s")
        return pulsar.luminosity_spindown(t).value

    for t_ in time:
        energy = scipy.integrate.quad(lumi, 0, t_.cgs.value, epsrel=0.01)[0]
        energies.append(energy)
    # The last value is quite inaccurate, because integration is over several decades
    assert_allclose(energies, pulsar.energy_integrated(time).value, rtol=0.2)


def test_Pulsar_magnetic_field():
    b = pulsar.magnetic_field(time)
    assert b.unit == "G"
    assert pulsar.B.unit == "G"
    assert_allclose(b.value, pulsar.B.value)

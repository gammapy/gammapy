# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.utils.data import get_pkg_data_filename
from ...utils.testing import assert_quantity
from ...irf import TablePSF, EnergyDependentTablePSF
from ...datasets import FermiGalacticCenter

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_TablePSF_gauss():
    # Make an example PSF for testing
    width = Angle(0.3, 'deg')
    offset = Angle(np.linspace(0, 2.3, 1000), 'deg')
    psf = TablePSF.from_shape(shape='gauss', width=width, offset=offset)

    offset = Angle([0.1, 0.3], 'deg')

    assert_allclose(psf.integral(), 1, rtol=1e-3)


@pytest.mark.skipif('not HAS_SCIPY')
def test_TablePSF_disk():

    width = Angle(2, 'deg')
    offset = Angle(np.linspace(0, 2.3, 1000), 'deg')
    psf = TablePSF.from_shape(shape='disk', width=width, offset=offset)

    # Check psf.eval by checking if probabilities sum to 1
    psf_value = psf.eval(offset, quantity='dp_dtheta')
    integral = np.sum(np.diff(offset.radian) * psf_value[:-1])
    assert_allclose(integral, 1, rtol=1e-3)

    psf_value = psf.eval(offset, quantity='dp_domega')
    psf_value = (2 * np.pi * offset * psf_value).to('radian^-1')
    integral = np.sum(np.diff(offset.radian) * psf_value[:-1])
    assert_allclose(integral, 1, rtol=1e-3)

    assert_allclose(psf.integral(), 1, rtol=1e-3)
    assert_allclose(psf.integral(*Angle([0, 10], 'deg')), 1, rtol=1e-3)
    assert_allclose(psf.integral(*Angle([0, 1], 'deg')), 0.25, rtol=1e-4)
    assert_allclose(psf.integral(*Angle([1, 2], 'deg')), 0.75, rtol=1e-2)

    # TODO
    #actual = psf.containment_radius([0.01, 0.25, 0.99])
    #desired = Angle([0, 1, 2], 'deg')
    #assert_quantity(actual, desired, rtol=1e-3)


@pytest.mark.skipif('not HAS_SCIPY')
def test_TablePSF():

    # Make an example PSF for testing
    width = Angle(0.3, 'deg')
    offset = Angle(np.linspace(0, 2.3, 1000), 'deg')
    psf = TablePSF.from_shape(shape='gauss', width=width, offset=offset)

    # Test inputs
    offset = Angle([0.1, 0.3], 'deg')

    actual = psf.eval(offset=offset, quantity='dp_domega')
    desired = Quantity([5491.52067694, 3521.07804604], 'sr^-1')
    assert_quantity(actual, desired)

    actual = psf.eval(offset=offset, quantity='dp_dtheta')
    desired = Quantity([60.22039017, 115.83738017], 'rad^-1')
    assert_quantity(actual, desired, rtol=1e-6)

    offset_min = Angle([0.0, 0.1, 0.3], 'deg')
    offset_max = Angle([0.1, 0.3, 2.0], 'deg')
    actual = psf.integral(offset_min, offset_max)
    desired = [0.05403975, 0.33942469, 0.60653066]
    assert_allclose(actual, desired)


@pytest.mark.skipif('not HAS_SCIPY')
def test_EnergyDependentTablePSF():

    # TODO: test __init__

    filename = FermiGalacticCenter.filenames()['psf']
    psf = EnergyDependentTablePSF.read(filename)

    # Test cases
    energy = Quantity(1, 'GeV')
    offset = Angle(0.1, 'deg')
    energies = Quantity([1, 2], 'GeV').to('TeV')
    offsets = Angle([0.1, 0.2], 'deg')
    
    pixel_size = Angle(0.1, 'deg')

    #actual = psf.eval(energy=energy, offset=offset)
    #desired = Quantity(17760.814249206363, 'sr^-1')
    #assert_quantity(actual, desired)

    #actual = psf.eval(energy=energies, offset=offsets)
    #desired = Quantity([17760.81424921, 5134.17706619], 'sr^-1')
    #assert_quantity(actual, desired)

    psf1 = psf.table_psf_at_energy(energy)

    # TODO: test average_psf
    #psf2 = psf.psf_in_energy_band(energy_band, spectrum)

    # TODO: test containment_radius
    # TODO: test containment_fraction
    # TODO: test info
    # TODO: test plotting methods

    desired = 1.0
    energy_band = Quantity([10, 500], 'GeV')
    psf_band = psf.table_psf_in_energy_band(energy_band)
    actual = psf_band.kernel(pixel_size, pixel_size, normalize=True).value.sum()

    assert_allclose(actual, desired)


def interactive_test():
    filename = get_pkg_data_filename('../../datasets/fermi/psf.fits')
    psf = EnergyDependentTablePSF.read(filename)
    # psf.plot_containment('fermi_psf_containment.pdf')
    # psf.plot_exposure('fermi_psf_exposure.pdf')
    psf.plot_theta('fermi_psf_theta.pdf')

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from astropy.units import Quantity, Unit
from astropy.utils.data import get_pkg_data_filename
from ..psf_table import TablePSF, EnergyDependentTablePSF, make_table_psf


def test_TablePSF():

    width = Quantity(0.2, 'deg')
    offset = Quantity(np.linspace(0, 0.7, 100), 'deg')
    psf = make_table_psf(shape='gauss', width=width, offset=offset)

    # Test cases
    offset = Quantity(0.1, 'deg')
    offsets = Quantity([0.1, 0.2], 'deg').to('rad')

    # FIXME: I think make_table_psf doesn't work properly yet
    # ... check units and that it integrates to 1.
    psf_eval = psf.eval(offset=offset)
    assert_allclose(psf_eval, 3.4569548710439073)
    assert psf_eval.unit == Unit('sr^-1')

    psf_eval = psf.eval(offset=offsets)
    assert_allclose(psf_eval, [3.45695487, 2.35237954])
    assert psf_eval.unit == Unit('sr^-1')


def test_EnergyDependentTablePSF():

    # TODO: test __init__

    filename = get_pkg_data_filename('../../datasets/fermi/psf.fits')
    psf = EnergyDependentTablePSF.read(filename)

    # Test cases
    energy = Quantity(1, 'GeV')
    offset = Quantity(0.1, 'deg')
    energies = Quantity([1, 2], 'GeV').to('TeV')
    offsets = Quantity([0.1, 0.2], 'deg').to('rad')

    psf_eval = psf.eval(energy=energy, offset=offset)
    assert_allclose(psf_eval, 17760.814249206363)
    assert psf_eval.unit == Unit('sr^-1')

    psf_eval = psf.eval(energy=energies, offset=offsets)
    assert_allclose(psf_eval, [17760.81424921, 5134.17706619])
    assert psf_eval.unit == Unit('sr^-1')

    # TODO: test average_psf
    # TODO: test containment_radius
    # TODO: test containment_fraction
    # TODO: test info
    # TODO: test plotting methods


"""
    # Create a TablePSF from an EnergyDependentTablePSF for Fermi    
    filename = get_pkg_data_filename('../../datasets/fermi/psf.fits')
    psf = EnergyDependentTablePSF.read(filename)
    energy = Quantity(1, 'GeV')
    psf = psf.table_psf(energy)
"""

def interactive_test():
    filename = get_pkg_data_filename('../../datasets/fermi/psf.fits')
    psf = EnergyDependentTablePSF.read(filename)
    # psf.plot_containment('fermi_psf_containment.pdf')
    # psf.plot_exposure('fermi_psf_exposure.pdf')
    psf.plot_theta('fermi_psf_theta.pdf')

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.tests.helper import assert_quantity_allclose
from astropy.table import Table
import numpy as np
from ...irf import PSFKing
from ...utils.energy import Energy
from ...utils.testing import requires_data
from ...datasets import gammapy_extra


@requires_data('gammapy-extra')
def test_psf_king_read():
    filename = str(gammapy_extra.dir) + "/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523" \
                                        "/hess_psf_king_023523.fits.gz"
    psf_king = PSFKing.read(filename)
    table = Table.read(filename)
    elo = table["ENERG_LO"].squeeze()
    ehi = table["ENERG_HI"].squeeze()
    offlo = table["THETA_LO"].squeeze()
    offhi = table["THETA_HI"].squeeze()
    sigma = table["SIGMA"].squeeze()
    gamma = table["GAMMA"].squeeze()
    energy = np.sqrt(elo * ehi)
    offset = (offlo + offhi) / 2.
    offset = Angle(offset, unit=table['THETA_LO'].unit)
    energy = Energy(energy, unit=table['ENERG_LO'].unit)
    gamma = Quantity(gamma, table['GAMMA'].unit)
    sigma = Quantity(sigma, table['SIGMA'].unit)

    assert_quantity_allclose(energy, psf_king.energy)
    assert_quantity_allclose(offset, psf_king.offset)
    assert_quantity_allclose(gamma, psf_king.gamma)
    assert_quantity_allclose(sigma, psf_king.sigma)


@requires_data('gammapy-extra')
def test_psf_king_write():
    filename = str(gammapy_extra.dir) + "/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523" \
                                        "/hess_psf_king_023523.fits.gz"
    psf_king = PSFKing.read(filename)
    psf_king.write("king.fits")
    psf_king2 = PSFKing.read("king.fits")
    assert_quantity_allclose(psf_king2.energy, psf_king.energy)
    assert_quantity_allclose(psf_king2.offset, psf_king.offset)
    assert_quantity_allclose(psf_king2.gamma, psf_king.gamma)
    assert_quantity_allclose(psf_king2.sigma, psf_king.sigma)


@requires_data('gammapy-extra')
def test_psf_king_evaluate():
    filename = str(gammapy_extra.dir) + "/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523" \
                                        "/hess_psf_king_023523.fits.gz"
    psf_king = PSFKing.read(filename)
    energy = Quantity(1, "TeV")
    off1 = Angle(0, "deg")
    off2 = Angle(1, "deg")
    param_off1 = psf_king.evaluate(energy, off1)
    param_off2 = psf_king.evaluate(energy, off2)

    assert_quantity_allclose(param_off1["gamma"], psf_king.gamma[0, 8])
    assert_quantity_allclose(param_off2["gamma"], psf_king.gamma[2, 8])
    assert_quantity_allclose(param_off1["sigma"], psf_king.sigma[0, 8])
    assert_quantity_allclose(param_off2["sigma"], psf_king.sigma[2, 8])


@requires_data('gammapy-extra')
def test_psf_king_to_table():
    filename = str(gammapy_extra.dir) + "/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523" \
                                        "/hess_psf_king_023523.fits.gz"
    psf_king = PSFKing.read(filename)
    theta1 = Angle(0, "deg")
    theta2 = Angle(1, "deg")
    psf_king_table_off1 = psf_king.to_table_psf(theta=theta1)
    psf_king_table_off2 = psf_king.to_table_psf(theta=theta2)
    offset = Angle(1, "deg")
    # energy = Quantity(1, "TeV") match with bin number 8
    # offset equal 1 degre match with the bin 200 in the psf_table
    value_off1 = psf_king.evaluate_direct(offset, psf_king.gamma[0, 8], psf_king.sigma[0, 8])
    value_off2 = psf_king.evaluate_direct(offset, psf_king.gamma[2, 8], psf_king.sigma[2, 8])
    # Test that the value at 1 degree in the histogram for the energy 1 Tev and theta=0 or 1 degree is equal to the one
    # obtained from the self.evaluate_direct() method at 1 degree
    assert_quantity_allclose(psf_king_table_off1.psf_value[8, 200], value_off1)
    assert_quantity_allclose(psf_king_table_off2.psf_value[8, 200], value_off2)

    # Test that the integral value is close to one
    bin_off = (psf_king_table_off1.offset[1] - psf_king_table_off1.offset[0])
    int = np.sum(psf_king_table_off1.psf_value[8] * 2 * np.pi * psf_king_table_off1.offset * bin_off)
    assert_quantity_allclose(int, 1, rtol=1e-1)

# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.units import Quantity
from astropy.coordinates import Angle
from numpy.testing import assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from astropy.table import Table
import numpy as np
from ...irf import PSFKing
from ...utils.energy import Energy
from ...utils.testing import requires_dependency, requires_data
from ...datasets import gammapy_extra


@requires_data('gammapy-extra')
def test_psf_king_read():
    filename = str(gammapy_extra.dir)+"/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523" \
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
    filename = str(gammapy_extra.dir)+"/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523" \
                                      "/hess_psf_king_023523.fits.gz"
    psf_king = PSFKing.read(filename)
    psf_king.write("king.fits")
    psf_king2=PSFKing.read("king.fits")
    assert_quantity_allclose(psf_king2.energy, psf_king.energy)
    assert_quantity_allclose(psf_king2.offset, psf_king.offset)
    assert_quantity_allclose(psf_king2.gamma, psf_king.gamma)
    assert_quantity_allclose(psf_king2.sigma, psf_king.sigma)

@requires_data('gammapy-extra')
def test_psf_king_evaluate():
    filename = str(gammapy_extra.dir)+"/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523" \
                                      "/hess_psf_king_023523.fits.gz"
    psf_king = PSFKing.read(filename)
    energy=Quantity(1, "TeV")
    off1=Angle(0, "deg")
    off2=Angle(1, "deg")
    param_off1 = psf_king.evaluate(energy, off1)
    param_off2 = psf_king.evaluate(energy, off2)

    assert_quantity_allclose(param_off1["gamma"], psf_king.gamma[0,8])
    assert_quantity_allclose(param_off2["gamma"], psf_king.gamma[2,8])
    assert_quantity_allclose(param_off1["sigma"], psf_king.sigma[0,8])
    assert_quantity_allclose(param_off2["sigma"], psf_king.sigma[2,8])

@requires_data('gammapy-extra')
def test_psf_king_to_table():
    filename = str(gammapy_extra.dir)+"/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523" \
                                      "/hess_psf_king_023523.fits.gz"
    psf_king = PSFKing.read(filename)
    energy=Quantity(1, "TeV")
    off1=Angle(0, "deg")
    off2=Angle(1, "deg")
    param_off1 = psf_king.evaluate(energy, off1)
    param_off2 = psf_king.evaluate(energy, off2)

    assert_quantity_allclose(param_off1["gamma"], psf_king.gamma[0,8])
    assert_quantity_allclose(param_off2["gamma"], psf_king.gamma[2,8])
    assert_quantity_allclose(param_off1["sigma"], psf_king.sigma[0,8])
    assert_quantity_allclose(param_off2["sigma"], psf_king.sigma[2,8])


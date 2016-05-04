# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.units import Quantity
from astropy.coordinates import Angle
from numpy.testing import assert_allclose
from astropy.table import Table
import numpy as np
from ...irf import PSFKing
from ...utils.energy import Energy
from ...utils.testing import requires_dependency, requires_data


@requires_data('gammapy-extra')
@requires_dependency('scipy')
def test_psf_king_read():
    filename="/Users/jouvin/Desktop/these/test_Gammapy/gammapy-extra/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_psf_king_023523.fits.gz"
    #filename = "$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_psf_king_023523.fits.gz"
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


    assert_allclose(energy, psf_king.energy)
    assert_allclose(offset, psf_king.offset)
    assert_allclose(gamma, psf_king.gamma)
    assert_allclose(sigma, psf_king.sigma)

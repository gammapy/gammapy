# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import unittest
from numpy.testing import assert_almost_equal
import numpy as np
from ..psf import HESS


class TestHESS(unittest.TestCase):
    def test_dpdtheta2(self):
        """Check that the amplitudes and sigmas were converted correctly in
        HESS.to_MultiGauss2D() by comparing the dpdtheta2 distribution.

        Note that we set normalize=False in the to_MultiGauss2D call,
        which is necessary because the HESS PSF is *not* normalized
        correcly by the HESS software, it is usually a few % off.

        Also quite interesting is to look at the norms, since they
        represent the fractions of gammas in each of the three components.

        integral: 0.981723
        sigmas:   [ 0.0219206   0.0905762   0.0426358]
        norms:    [ 0.29085818  0.20162012  0.48924452]

        So in this case the HESS PSF 'scale' is 2% too low
        and e.g. the wide sigma = 0.09 deg PSF component contains 20%
        of the events.
        """
        hess = HESS('input/gc_psf.txt')
        m = hess.to_MultiGauss2D(normalize=False)
        if 0:
            print('integral:', m.integral)
            print('sigmas:  ', m.sigmas)
            print('norms:   ', m.norms)

        for theta in np.linspace(0, 1, 10):
            val_hess = hess.dpdtheta2(theta ** 2)
            val_m = m.dpdtheta2(theta ** 2)
            assert_almost_equal(val_hess, val_m, decimal=4)

    def test_GC(self):
        """Compare the containment radii computed with the HESS software
        with those found by using MultiGauss2D.

        This test fails for r95, where the HESS software gives a theta
        which is 10% higher. Probably the triple-Gauss doesn't represent
        the PSF will in the core or the fitting was bad or the
        HESS software has very large binning errors (they compute
        containment radius from the theta2 histogram directly, not
        using the triple-Gauss approximation)."""
        vals = [(68, 0.0663391),
                (95, 0.173846),  # 0.15310963243226974
                (10, 0.0162602),
                (40, 0.0379536),
                (80, 0.088608)]
        hess = HESS('input/gc_psf.txt')
        m = hess.to_MultiGauss2D()
        assert_almost_equal(m.integral, 1)
        for containment, theta in vals:
            assert_almost_equal(m.theta(containment / 100.), theta, decimal=2)

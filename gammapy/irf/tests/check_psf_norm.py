# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.tests.helper import pytest


@pytest.mark.xfail
def test_psf_norm():
    from ROOT import TF1

    # Triple exponential 1D pdf in x = theta ^ 2
    # represents a triple Gaussian 2D pdf
    _psf = "[0]*(exp(-x/(2*[1]*[1]))+[2]*exp(-x/(2*[3]*[3]))+[4]*exp(-x/(2*[5]*[5])))"
    theta2_min, theta2_max = 0.00, 0.05
    psf = TF1("psf", _psf, theta2_min, theta2_max)
    psf.SetParNames("SCALE", "SIGMA_1", "AMPL_2", "SIGMA_2", "AMPL_3", "SIGMA_3")

    # First entry from Karl's file CTA1DC-HESS-run00023523_std_psf.fits
    # Invalid values (probably fit did not converge)
    psf.SetParameters(177.319000000000, 0.0272624000000000, 0.00641271000000000,
                      0.999995000000000, 0.171255000000000, 1.00000000000000)
    print('Karl first: ', psf.Integral(theta2_min, theta2_max))

    # Last entry from Karl's file CTA1DC-HESS-run00023523_std_psf.fits
    psf.SetParameters(500.548000000000, 0.0145670000000000, 0.00489170000000000,
                      0.121316000000000, 0.279361000000000, 0.0337978000000000)
    print('Karl last: ', psf.Integral(theta2_min, theta2_max))

    # Some other example of HESS PSF by Christoph
    psf.SetParameters(285.117, 0.021611, 0.0477598,
                      0.0919602, 0.461796, 0.0426815)
    print('Christoph: ', psf.Integral(theta2_min, theta2_max))

    # === Output ===
    # Karl first:  1.81924677453
    # Karl last:  0.590779978532
    # Christoph:  0.964365136188
